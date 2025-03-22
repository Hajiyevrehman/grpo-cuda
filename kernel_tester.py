import torch
import torch.nn as nn
import os
import time
import numpy as np
import random
from dataclasses import dataclass, field
import importlib.util
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

@dataclass
class KernelExecResult:
    """Result of kernel execution evaluation"""
    compiled: bool = False
    correctness: bool = False
    metadata: dict = field(default_factory=dict)
    runtime: float = -1.0  # in ms
    runtime_stats: dict = field(default_factory=dict)

def evaluate_kernel(ref_code, kernel_code, build_dir, num_correct_trials=5, num_perf_trials=10):
    """
    Evaluate a kernel against reference implementation
    
    Args:
        ref_code: PyTorch reference code
        kernel_code: Custom kernel code
        build_dir: Directory to build kernel
        num_correct_trials: Number of trials for correctness testing
        num_perf_trials: Number of trials for performance testing
        
    Returns:
        KernelExecResult object
    """
    # Ensure CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, cannot evaluate kernels")
    
    device = torch.cuda.current_device()
    metadata = {
        "hardware": torch.cuda.get_device_name(device=device),
        "device": str(device)
    }
    
    # Step 1: Load reference model
    print("Loading reference model...")
    ref_model, get_init_inputs, get_inputs = load_model_from_code(ref_code)
    if not all([ref_model, get_init_inputs, get_inputs]):
        return KernelExecResult(compiled=False, metadata=metadata)
    
    # Step 2: Compile and load kernel model
    print("Compiling and loading kernel model...")
    try:
        kernel_model = compile_and_load_kernel(kernel_code, build_dir)
        if kernel_model is None:
            metadata["compilation_error"] = "Failed to compile kernel"
            return KernelExecResult(compiled=False, metadata=metadata)
    except Exception as e:
        metadata["compilation_error"] = str(e)
        return KernelExecResult(compiled=False, metadata=metadata)
    
    # Step 3: Check correctness
    print("Checking correctness...")
    try:
        is_correct = check_correctness(
            ref_model, kernel_model, get_init_inputs, get_inputs, 
            num_correct_trials, device
        )
        if not is_correct:
            metadata["correctness_error"] = "Output mismatch"
            return KernelExecResult(compiled=True, correctness=False, metadata=metadata)
    except Exception as e:
        metadata["correctness_error"] = str(e)
        return KernelExecResult(compiled=True, correctness=False, metadata=metadata)
    
    # Step 4: Measure performance
    print("Measuring performance...")
    ref_times = measure_performance(ref_model, get_inputs, num_perf_trials, device)
    kernel_times = measure_performance(kernel_model, get_inputs, num_perf_trials, device)
    
    # Step 5: Calculate statistics
    ref_stats = get_timing_stats(ref_times)
    kernel_stats = get_timing_stats(kernel_times)
    
    # Calculate speedup
    speedup = ref_stats["mean"] / kernel_stats["mean"]
    metadata["speedup"] = speedup
    metadata["ref_stats"] = ref_stats
    
    return KernelExecResult(
        compiled=True, 
        correctness=True, 
        metadata=metadata,
        runtime=kernel_stats["mean"],
        runtime_stats=kernel_stats
    )

def load_model_from_code(model_src):
    """
    Load model class from source code string
    
    Returns:
        tuple: (Model class, get_init_inputs function, get_inputs function)
    """
    context = {}
    try:
        # Execute the model source code in the context
        exec(model_src, context)
        
        # Extract the model class and input functions
        Model = context.get("Model")
        get_init_inputs = context.get("get_init_inputs")
        get_inputs = context.get("get_inputs")
        
        if not all([Model, get_init_inputs, get_inputs]):
            print("Error: Model code must define 'Model', 'get_init_inputs', and 'get_inputs'")
            return None, None, None
            
        return Model, get_init_inputs, get_inputs
    except Exception as e:
        print(f"Error loading model from code: {e}")
        return None, None, None

def compile_and_load_kernel(kernel_code, build_dir):
    """
    Compile and load custom kernel code
    
    Args:
        kernel_code: Custom kernel source code
        build_dir: Directory to build the kernel
        
    Returns:
        ModelNew class or None if compilation fails
    """
    context = {}
    
    # Setup build directory in environment
    os.makedirs(build_dir, exist_ok=True)
    os.environ['TORCH_EXTENSIONS_DIR'] = build_dir
    
    # Set CUDA flags for device side assertions
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    
    try:
        # Execute the kernel code in the context
        exec(kernel_code, context)
        
        # Extract the model class
        ModelNew = context.get("ModelNew")
        if ModelNew is None:
            print("Error: Kernel code must define 'ModelNew'")
            return None
            
        return ModelNew
    except Exception as e:
        print(f"Error compiling kernel: {e}")
        return None

def check_correctness(Model, ModelNew, get_init_inputs, get_inputs, num_trials, device):
    """
    Check if the kernel implementation matches the reference
    
    Args:
        Model: Reference model class
        ModelNew: Custom kernel model class
        get_init_inputs: Function to get model initialization inputs
        get_inputs: Function to get model forward inputs
        num_trials: Number of trials with different inputs
        device: CUDA device
        
    Returns:
        bool: True if kernel is correct, False otherwise
    """
    # Generate seeds for trials
    torch.manual_seed(42)
    trial_seeds = [torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_trials)]
    
    with torch.no_grad():
        for trial, seed in enumerate(trial_seeds):
            # Set seed for reproducibility
            set_seed(seed)
            
            # Get initialization inputs
            init_inputs = get_init_inputs()
            init_inputs = [x.cuda(device) if isinstance(x, torch.Tensor) else x for x in init_inputs]
            
            # Initialize models
            set_seed(seed)
            ref_model = Model(*init_inputs).cuda(device)
            
            set_seed(seed)
            new_model = ModelNew(*init_inputs).cuda(device)
            
            # Get forward inputs
            set_seed(seed)
            inputs = get_inputs()
            inputs = [x.cuda(device) if isinstance(x, torch.Tensor) else x for x in inputs]
            
            # Run models
            ref_output = ref_model(*inputs)
            torch.cuda.synchronize(device)
            
            try:
                new_output = new_model(*inputs)
                torch.cuda.synchronize(device)
                
                # Check output shapes
                if ref_output.shape != new_output.shape:
                    print(f"Output shape mismatch: Expected {ref_output.shape}, got {new_output.shape}")
                    return False
                
                # Check output values
                if not torch.allclose(ref_output, new_output, atol=1e-2, rtol=1e-2):
                    max_diff = torch.max(torch.abs(ref_output - new_output)).item()
                    avg_diff = torch.mean(torch.abs(ref_output - new_output)).item()
                    print(f"Output mismatch: max_diff={max_diff:.6f}, avg_diff={avg_diff:.6f}")
                    return False
                
                print(f"Trial {trial}: Passed correctness check")
                
            except Exception as e:
                print(f"Error in kernel execution: {e}")
                return False
                
    return True

def measure_performance(model, get_inputs_fn, num_trials, device):
    """
    Measure model performance using CUDA events
    
    Args:
        model: Model to measure
        get_inputs_fn: Function to get model inputs
        num_trials: Number of timing trials
        device: CUDA device
        
    Returns:
        list: List of execution times in milliseconds
    """
    # Warm up runs
    set_seed(42)
    inputs = get_inputs_fn()
    inputs = [x.cuda(device) if isinstance(x, torch.Tensor) else x for x in inputs]
    
    # Warm up
    for _ in range(3):
        model(*inputs)
        torch.cuda.synchronize(device)
    
    # Timing runs
    elapsed_times = []
    for _ in range(num_trials):
        # Create CUDA events for timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Record start time
        start_event.record()
        
        # Run model
        model(*inputs)
        
        # Record end time
        end_event.record()
        
        # Synchronize and measure
        torch.cuda.synchronize(device)
        elapsed_time = start_event.elapsed_time(end_event)
        elapsed_times.append(elapsed_time)
    
    return elapsed_times

def get_timing_stats(elapsed_times):
    """
    Calculate timing statistics
    
    Args:
        elapsed_times: List of execution times in milliseconds
        
    Returns:
        dict: Dictionary of timing statistics
    """
    import numpy as np
    
    return {
        "mean": float(f"{np.mean(elapsed_times):.3g}"),
        "std": float(f"{np.std(elapsed_times):.3g}"),
        "min": float(f"{np.min(elapsed_times):.3g}"),
        "max": float(f"{np.max(elapsed_times):.3g}"),
        "num_trials": len(elapsed_times),
    }

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False