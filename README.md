This is research-code for 
[Synthesizing Obama: Learning Lip Sync from Audio.](grail.cs.washington.edu/projects/AudioToObama/)<br>
Supasorn Suwajanakorn, Steven M. Seitz, Ira Kemelmacher-Shlizerman<br>
SIGGRAPH 2017

Code tested using tensorflow 0.11.0
Please see [Supasorn's website](http://homes.cs.washington.edu/~supasorn/?page=code) for the overview.

To generate MFCC, first normalize the input audio using https://github.com/slhck/ffmpeg-normalize. Then use [Sphinx III's snippet](http://www.cs.cmu.edu/~dhuggins/Projects/pyphone/sphinx/mfcc.py) by David Huggins-Daines with a modified routine that saves log energy and timestamps:

```python
def sig2s2mfc_energy(self, sig, dn):
  nfr = int(len(sig) / self.fshift + 1)

  mfcc = numpy.zeros((nfr, self.ncep + 2), 'd')
  fr = 0
  while fr < nfr:
    start = int(round(fr * self.fshift))
    end = min(len(sig), start + self.wlen)
    frame = sig[start:end]
    if len(frame) < self.wlen:
      frame = numpy.resize(frame,self.wlen)
      frame[self.wlen:] = 0
    mfcc[fr,:-2] = self.frame2s2mfc(frame)
    mfcc[fr, -2] = math.log(1 + np.mean(np.power(frame.astype(float), 2)))
    mid = 0.5 * (start + end - 1)
    mfcc[fr, -1] = mid / self.samprate

    fr = fr + 1
  return mfcc
```

