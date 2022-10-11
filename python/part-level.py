import arviz
import pymc3
import matplotlib.figure as figure
import numpy

def idM(n):
  return numpy.diag(numpy.ones(n))


data = numpy.array([100, 225, 420])
datavars = numpy.array([[111, 226, 420], [100, 240, 400], [100, 225, 440]])
datavardiffs = datavars - data

nbins = data.shape[0]
nvariations = datavars.shape[0]

datacov = numpy.matmul(datavardiffs, datavardiffs.T)

background = numpy.array([100, 225, 400])
signal = numpy.array([0, 0, 20])

model = pymc3.Model()

with model:
  signalmu = pymc3.Uniform("signalmu", lower=-1000, upper=1000, shape=(1,))

  bkgnorm = pymc3.Normal("bkgnorm", mu=1, sigma=0.05, shape=(1,))

  predictions = background*bkgnorm + signalmu*signal

  _ = pymc3.Deterministic("predictions", predictions)

  pymc3.MvNormal \
    ( "llh"
    , mu = data
    , cov = datacov
    , observed = predictions
    , shape = (nbins,)
    )


with model:
  trace = \
    pymc3.sample \
    ( 10000
    , tune=2000
    , chains=4
    , return_inferencedata=True
    , cores=1
    )


fig = figure.Figure(figsize=(10, 10))
axes = fig.subplots(1, 2, squeeze=False)
arviz.plot_trace(trace, var_names=["predictions"], axes=axes)
fig.savefig("predictions.png")
fig.clf()

axes = fig.subplots(2, 2, squeeze=False)
arviz.plot_pair(trace, var_names=["predictions"], ax=axes)
fig.savefig("predictions-2d.png")
fig.clf()

axes = fig.subplots(1, 2, squeeze=False)
arviz.plot_trace(trace, var_names=["signalmu"], axes=axes)
fig.savefig("signalmu.png")
