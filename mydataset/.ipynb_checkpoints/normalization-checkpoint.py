import numpy as np

def robust_normalization(data, iqr_factor=0.5, axis=None):
    # Compute 25th and 75th percentiles
    q25, q75 = np.percentile(data, [25, 75], axis=axis, keepdims=True)
    iqr = q75 - q25
    
    lower_bound = q25 - iqr_factor * iqr
    upper_bound = q75 + iqr_factor * iqr
    
    # Filter data within bounds
    filtered_data = np.where((data >= lower_bound) & (data <= upper_bound), data, np.nan)
    nan_ratio = np.sum(np.isnan(filtered_data)) / data.size
    print("Out of Range Ratio:  %.8f %%"%(nan_ratio*100))

    min_val = np.nanmin(filtered_data, axis=axis, keepdims=True)
    max_val = np.nanmax(filtered_data, axis=axis, keepdims=True)
    
    # Compute mean and scale based on min/max range
    range_val = max_val - min_val
    
    var_mean = np.nanmean(filtered_data, axis=axis, keepdims=True)
    
    var_scale = range_val/2
    
    return var_mean, var_scale


def limit_range_log(x, thres = 10):
    x = np.copy(x)
    outlier_idx = (x < -1*thres) | (x > thres)
    outlier_ratio = np.sum(outlier_idx)/outlier_idx.size
    print("log2 limit scale ratio: %.4f %%"%(outlier_ratio*100))
    outlier = x[outlier_idx]
    x[outlier_idx] = (np.log2(np.abs(outlier)-(thres-1)) + thres)*np.sign(outlier)
    return x



def normalize_data(data, norm_type, axis):
    
    if norm_type == "std":
        var_mean = np.mean(data, axis = axis, keepdims = True)
        var_scale = np.std(data, axis = axis, keepdims = True)
        data = (data - var_mean)/var_scale
    elif norm_type == "range2":
        
        var_min = np.min(data, axis = axis, keepdims = True)
        data = data - var_min
        
        var_max = np.max(data, axis = axis, keepdims = True)
        data = data/(var_max/2) - 1
        var_mean, var_scale = (var_min+var_max/2), var_max/2
        
    elif norm_type == "range":
        var_mean = np.mean(data, axis = axis, keepdims = True)
        
        var_max = np.max(data, axis = axis, keepdims = True)
        var_min = np.min(data, axis = axis, keepdims = True)
        var_scale = var_max - var_min
        data = (data - var_mean)/var_scale
        
    elif norm_type == "robust":
        var_mean, var_scale = robust_normalization(data, 2, axis)
        data = (data - var_mean)/var_scale
        data = limit_range_log(data, 1)
        
    elif norm_type == "robust_nolimit":
        var_mean, var_scale = robust_normalization(data, 2, axis)
        data = (data - var_mean)/var_scale
    else:
        assert NoImplementError
        
        
    return data, var_mean, var_scale
        