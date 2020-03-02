import numpy as np
from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.classifiers.keras import KerasClassifier

def adv(x, dnn):
