def sign_ae(x, y):
    import theano.tensor as T
    sign_x = T.sgn(x)
    sign_y = T.sgn(y)
    delta = x - y
    return sign_x * sign_y * T.abs_(delta)
    
    
def linex_loss(delta, a=-1, b=1):
    import theano.tensor as T
    if a!= 0 and b > 0:
        loss = b * (T.exp(a * delta) - a * delta - 1)
        return loss
    else:
        raise ValueError
        
        
def linex_loss_val(y_true, y_pred):
    import theano.tensor as T   
    delta = sign_ae(y_true, y_pred)
    res = linex_loss(delta)
    return T.mean(res)
    
    
def linex_loss_ret(y_true, y_pred):
    import theano.tensor as T
    diff_true = T.extra_ops.diff(y_true, axis=0)
    diff_pred = T.extra_ops.diff(y_pred, axis=0)
    
    delta = sign_ae(diff_true, diff_pred)
    res = linex_loss(delta)
    return T.mean(res)
