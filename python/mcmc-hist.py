import arviz
import pymc3
import matplotlib.figure as figure
import numpy

model = pymc3.Model()

data = numpy.array([100, 200, 400])
nbins = data.shape[0]

nominal = numpy.array([120, 180, 300])
variations = [ numpy.array([100, 200, 400]) ]
vardiffs = variations - nominal
nvariations = len(variations)


with model:
  theta = \
    pymc3.MvNormal \
    ( "theta"
    , mu=numpy.zeros(nvariations)
    , cov=numpy.diag(numpy.ones(nvariations))
    , shape=(nvariations,)
    )

  predictions = nominal
  for i in range(nvariations):
    predictions += theta[i] * vardiffs[i]

  binllhs = []
  for i in range(nbins):
    binllhs.append \
      ( pymc3.Poisson("poisson%03d" % i, mu=predictions[i], observed=data[i])
      )

  pymc3.math.sum(binllhs)



with model:
  trace = \
    pymc3.sample \
    ( 10000
    , tune=2000
    , chains=4
    , return_inferencedata=True
    , cores=1
    )

fig = figure.Figure()
plt1 = fig.add_subplot( 1 , 2 , 1 )
plt2 = fig.add_subplot( 1 , 2 , 2 )


arviz.plot_trace(trace, var_names=["theta"], axes=numpy.array([[plt1 , plt2]]))

fig.savefig("test.pdf")
