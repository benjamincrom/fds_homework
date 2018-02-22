import numpy


def l_beta(X, Y, beta):
    return sum(
        [
            numpy.log(
                (1 / (1 + numpy.exp(-1 * numpy.dot(numpy.transpose(beta), X[i]))))**Y[i] *
                (1 - (1 / (1 + numpy.exp(-1 * numpy.dot(numpy.transpose(beta), X[i])))))**(1 - Y[i])
            )
            for i in range(len(X))
        ]
    )


def gradient_l_beta(X, Y, beta):
    return sum(
        [
            (Y[i] - (1 / (1 + numpy.exp(-1 * numpy.dot(numpy.transpose(beta), X[i]))))) * X[i]
            for i in range(len(X))
        ]
    )


def gradient_ascent(X, Y, beta, eta, tol):
    last_l_beta = l_beta(X, Y, beta)
    beta = beta + (eta * gradient_l_beta(X, Y, beta))
    new_l_beta = l_beta(X, Y, beta)

    while numpy.absolute(new_l_beta - last_l_beta) > tol:
        last_l_beta = new_l_beta
        beta = beta + (eta * gradient_l_beta(X, Y, beta))
        new_l_beta = l_beta(X, Y, beta)
        
    return beta, l_beta(X, Y, beta)


def classify_logReg(X, beta):
    Y_hat = []
    for i in range(len(X)):
        y_1_val = (1 / (1 + numpy.exp(-1 * numpy.dot(numpy.transpose(beta), X[i]))))
        y_0_val = (1 - (1 / (1 + numpy.exp(-1 * numpy.dot(numpy.transpose(beta), X[i])))))
        
        if y_1_val > y_0_val:
            Y_hat.append(1.0)
        else:
            Y_hat.append(0.0)
            
    return numpy.array(Y_hat)

