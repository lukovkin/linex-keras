# linex-keras
Theano-based implementation of asymmetric loss functions.

Traditionally used loss functions (MSE, MAE, MAPE) are symmetric in the mean that value of loss function is the same in the case of overestimation of X and in the case of underestimation. That is not applicable to the wide range of cases - e.g. you estimate time when you have to arrive to the airport to not miss the flight. In this case if you underestimate time by 10 minutes, you will just arrive a bit earlier, but if you overestimate by 10 minutes you will completely miss the plane.

So there is a range of asymmetrical loss functions which could produce values that differ for the positive and negative arguments of the same abdolute value.

Keras doesn't contain asymmetrical loss functions in it, so they are implemented separately.

LINEX - linear exponential loss function. Introduced by Varian (1975), Zellner (1986). See Bayesian Analysis in Statistics and Econometrics: Essays in Honor of Arnold Zellner, pp. 471-485 for details.

LINEX(x) = b * (exp(a * x) - a * x - 1), a != 0, b > 0

Use a < 0 if you want to penalize errors with negative values more than errors with positive values, and a > 0 otherwise.

