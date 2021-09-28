import  numpy as np

# return  values 0 or 1
def stepFunction(sum):
    if (sum >= 1):
        return 1
    return 0

# return values between 0 and 1
def sigmoidFunction(sum):
    return 1 / (1 + np.exp(-sum))

# return values between -1 and 1
def tahnFunction(sum):
    return (np.exp(sum) - np.exp(-sum)) / (np.exp(sum) + np.exp(-sum))

# return value 0 or bigger
def reluFunction(sum):
    if (sum >= 0):
        return sum
    return 0;

# return value passed
def linearFunction(sum):
    return sum

#
def softmaxFunction(x):
    ex = np.exp(x)
    return ex / ex.sum()

step = stepFunction(30);
sigmoid = sigmoidFunction(20)
hyper = tahnFunction(2.1);
relu = reluFunction(110)
liner = linearFunction(-110)

softmaxParams = [5.0, 2.0, 1.3]
print(softmaxFunction(softmaxParams))