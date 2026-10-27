"""Scalar executable equations for HISTORY-QUERY016 loss, not original network.
Source: TAC-OSM 3d7946a3e46233fc0c09e86f60ad2c24c27a1674.
IEEE/AD and whole-model formal equivalence OPEN. No model capability inferred.
"""
import math


def masked_mean(values, observed):
    return sum(v for v,m in zip(values,observed) if m)/max(1,sum(observed))


def historical_loss(pred, target, observed):
    def huber(e):
        return .5*e*e if abs(e)<1 else abs(e)-.5
    # Never evaluate absent targets (which may contain NaN).
    return masked_mean([huber(p-y) if m else 0. for p,y,m in zip(pred,target,observed)],observed)


def forecast_loss(mu, sd, target, observed):
    vals=[]
    for p,s,y,m in zip(mu,sd,target,observed):
        if s<=0:raise ValueError('positive forecast scale required')
        vals.append(.8*.5*(math.log(2*math.pi)+2*math.log(s)+((y-p)/s)**2)+.2*abs(y-p) if m else 0.)
    return masked_mean(vals,observed)
