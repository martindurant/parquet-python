# Copyright (c) 1999-2008 Gary Strangman; All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Comments and/or additions are welcome (send e-mail to:
# strang@nmr.mgh.harvard.edu).
#
"""
stats.py module

(Requires pstat.py module.)

#################################################
#######  Written by:  Gary Strangman  ###########
#######  Last modified:  Oct 31, 2008 ###########
#################################################

A collection of basic statistical functions for python.  The function
names appear below.

IMPORTANT:  There are really *3* sets of functions.  The first set has an 'l'
prefix, which can be used with list or tuple arguments.  The second set has
an 'a' prefix, which can accept NumPy array arguments.  These latter
functions are defined only when NumPy is available on the system.  The third
type has NO prefix (i.e., has the name that appears below).  Functions of
this set are members of a "Dispatch" class, c/o David Ascher.  This class
allows different functions to be called depending on the type of the passed
arguments.  Thus, stats.mean is a member of the Dispatch class and
stats.mean(range(20)) will call stats.lmean(range(20)) while
stats.mean(Numeric.arange(20)) will call stats.amean(Numeric.arange(20)).
This is a handy way to keep consistent function names when different
argument types require different functions to be called.  Having
implementated the Dispatch class, however, means that to get info on
a given function, you must use the REAL function name ... that is
"print stats.lmean.__doc__" or "print stats.amean.__doc__" work fine,
while "print stats.mean.__doc__" will print the doc for the Dispatch
class.  NUMPY FUNCTIONS ('a' prefix) generally have more argument options
but should otherwise be consistent with the corresponding list functions.

Disclaimers:  The function list is obviously incomplete and, worse, the
functions are not optimized.  All functions have been tested (some more
so than others), but they are far from bulletproof.  Thus, as with any
free software, no warranty or guarantee is expressed or implied. :-)  A
few extra functions that don't appear in the list below can be found by
interested treasure-hunters.  These functions don't necessarily have
both list and array versions but were deemed useful

CENTRAL TENDENCY:  geometricmean
                   harmonicmean
                   mean
                   median
                   medianscore
                   mode

MOMENTS:  moment
          variation
          skew
          kurtosis
          skewtest   (for Numpy arrays only)
          kurtosistest (for Numpy arrays only)
          normaltest (for Numpy arrays only)

ALTERED VERSIONS:  tmean  (for Numpy arrays only)
                   tvar   (for Numpy arrays only)
                   tmin   (for Numpy arrays only)
                   tmax   (for Numpy arrays only)
                   tstdev (for Numpy arrays only)
                   tsem   (for Numpy arrays only)
                   describe

FREQUENCY STATS:  itemfreq
                  scoreatpercentile
                  percentileofscore
                  histogram
                  cumfreq
                  relfreq

VARIABILITY:  obrientransform
              samplevar
              samplestdev
              signaltonoise (for Numpy arrays only)
              var
              stdev
              sterr
              sem
              z
              zs
              zmap (for Numpy arrays only)

TRIMMING FCNS:  threshold (for Numpy arrays only)
                trimboth
                trim1
                round (round all vals to 'n' decimals; Numpy only)

CORRELATION FCNS:  covariance  (for Numpy arrays only)
                   correlation (for Numpy arrays only)
                   paired
                   pearsonr
                   spearmanr
                   pointbiserialr
                   kendalltau
                   linregress

INFERENTIAL STATS:  ttest_1samp
                    ttest_ind
                    ttest_rel
                    chisquare
                    ks_2samp
                    mannwhitneyu
                    ranksums
                    wilcoxont
                    kruskalwallish
                    friedmanchisquare

PROBABILITY CALCS:  chisqprob
                    erfcc
                    zprob
                    ksprob
                    fprob
                    betacf
                    gammln
                    betai

ANOVA FUNCTIONS:  F_oneway
                  F_value

SUPPORT FUNCTIONS:  writecc
                    incr
                    sign  (for Numpy arrays only)
                    sum
                    cumsum
                    ss
                    summult
                    sumdiffsquared
                    square_of_sums
                    shellsort
                    rankdata
                    outputpairedstats
                    findwithin
"""
## CHANGE LOG:
## ===========
## 09-07-21 ... added capability for getting the 'proportion' out of l/amannwhitneyu (but comment-disabled)
## 08-10-31 ... fixed import LinearAlgebra bug before glm fcns
## 07-11-26 ... conversion for numpy started
## 07-05-16 ... added Lin's Concordance Correlation Coefficient (alincc) and acov
## 05-08-21 ... added "Dice's coefficient"
## 04-10-26 ... added ap2t(), an ugly fcn for converting p-vals to T-vals
## 04-04-03 ... added amasslinregress() function to do regression on N-D arrays
## 03-01-03 ... CHANGED VERSION TO 0.6
##              fixed atsem() to properly handle limits=None case
##              improved histogram and median functions (estbinwidth) and
##                   fixed atvar() function (wrong answers for neg numbers?!?)
## 02-11-19 ... fixed attest_ind and attest_rel for div-by-zero Overflows
## 02-05-10 ... fixed lchisqprob indentation (failed when df=even)
## 00-12-28 ... removed aanova() to separate module, fixed licensing to
##                   match Python License, fixed doc string & imports
## 00-04-13 ... pulled all "global" statements, except from aanova()
##              added/fixed lots of documentation, removed io.py dependency
##              changed to version 0.5
## 99-11-13 ... added asign() function
## 99-11-01 ... changed version to 0.4 ... enough incremental changes now
## 99-10-25 ... added acovariance and acorrelation functions
## 99-10-10 ... fixed askew/akurtosis to avoid divide-by-zero errors
##              added aglm function (crude, but will be improved)
## 99-10-04 ... upgraded acumsum, ass, asummult, asamplevar, avar, etc. to
##                   all handle lists of 'dimension's and keepdims
##              REMOVED ar0, ar2, ar3, ar4 and replaced them with around
##              reinserted fixes for abetai to avoid math overflows
## 99-09-05 ... rewrote achisqprob/aerfcc/aksprob/afprob/abetacf/abetai to
##                   handle multi-dimensional arrays (whew!)
## 99-08-30 ... fixed l/amoment, l/askew, l/akurtosis per D'Agostino (1990)
##              added anormaltest per same reference
##              re-wrote azprob to calc arrays of probs all at once
## 99-08-22 ... edited attest_ind printing section so arrays could be rounded
## 99-08-19 ... fixed amean and aharmonicmean for non-error(!) overflow on
##                   short/byte arrays (mean of # s btw 100-300 = -150??)
## 99-08-09 ... fixed asum so that the None case works for Byte arrays
## 99-08-08 ... fixed 7/3 'improvement' to handle t-calcs on N-D arrays
## 99-07-03 ... improved attest_ind, attest_rel (zero-division errortrap)
## 99-06-24 ... fixed bug(?) in attest_ind (n1=a.shape[0])
## 04/11/99 ... added asignaltonoise, athreshold functions, changed all
##                   max/min in array section to N.maximum/N.minimum,
##                   fixed square_of_sums to prevent integer overflow
## 04/10/99 ... !!! Changed function name ... sumsquared ==> square_of_sums
## 03/18/99 ... Added ar0, ar2, ar3 and ar4 rounding functions
## 02/28/99 ... Fixed aobrientransform to return an array rather than a list
## 01/15/99 ... Essentially ceased updating list-versions of functions (!!!)
## 01/13/99 ... CHANGED TO VERSION 0.3
##              fixed bug in a/lmannwhitneyu p-value calculation
## 12/31/98 ... fixed variable-name bug in ldescribe
## 12/19/98 ... fixed bug in findwithin (fcns needed pstat. prefix)
## 12/16/98 ... changed amedianscore to return float (not array) for 1 score
## 12/14/98 ... added atmin and atmax functions
##              removed umath from import line (not needed)
##              l/ageometricmean modified to reduce chance of overflows (take
##                   nth root first, then multiply)
## 12/07/98 ... added __version__variable (now 0.2)
##              removed all 'stats.' from anova() fcn
## 12/06/98 ... changed those functions (except shellsort) that altered
##                   arguments in-place ... cumsum, ranksort, ...
##              updated (and fixed some) doc-strings
## 12/01/98 ... added anova() function (requires NumPy)
##              incorporated Dispatch class
## 11/12/98 ... added functionality to amean, aharmonicmean, ageometricmean
##              added 'asum' function (added functionality to N.add.reduce)
##              fixed both moment and amoment (two errors)
##              changed name of skewness and askewness to skew and askew
##              fixed (a)histogram (which sometimes counted points <lowerlimit)

import math
import copy
# from types import *

import pstat


__version__ = 0.6

############# DISPATCH CODE ##############


####################################
#######  CENTRAL TENDENCY  #########
####################################

def geometricmean(inlist):
    """
Calculates the geometric mean of the values in the passed list.
That is:  n-th root of (x1 * x2 * ... * xn).  Assumes a '1D' list.

Usage:   lgeometricmean(inlist)
"""
    mult = 1.0
    one_over_n = 1.0 / len(inlist)
    for item in inlist:
        mult = mult * pow(item, one_over_n)
    return mult


def harmonicmean(inlist):
    """
Calculates the harmonic mean of the values in the passed list.
That is:  n / (1/x1 + 1/x2 + ... + 1/xn).  Assumes a '1D' list.

Usage:   lharmonicmean(inlist)
"""
    sum = 0
    for item in inlist:
        sum = sum + 1.0 / item
    return len(inlist) / sum


def mean(inlist):
    """
Returns the arithematic mean of the values in the passed list.
Assumes a '1D' list, but will function on the 1st dim of an array(!).

Usage:   lmean(inlist)
"""
    sum = 0
    for item in inlist:
        sum = sum + item
    return sum / float(len(inlist))


def median(inlist, numbins=1000):
    """
Returns the computed median value of a list of numbers, given the
number of bins to use for the histogram (more bins brings the computed value
closer to the median score, default number of bins = 1000).  See G.W.
Heiman's Basic Stats (1st Edition), or CRC Probability & Statistics.

Usage:   lmedian (inlist, numbins=1000)
"""
    (hist, smallest, binsize, extras) = histogram(inlist, numbins, [min(inlist), max(inlist)]) # make histog
    cumhist = cumsum(hist)              # make cumulative histogram
    for i in range(len(cumhist)):        # get 1st(!) index holding 50%ile score
        if cumhist[i] >= len(inlist) / 2.0:
            cfbin = i
            break
    LRL = smallest + binsize * cfbin        # get lower read limit of that bin
    cfbelow = cumhist[cfbin - 1]
    freq = float(hist[cfbin])                # frequency IN the 50%ile bin
    median = LRL + ((len(inlist) / 2.0 - cfbelow) / float(freq)) * binsize  # median formula
    return median


def medianscore(inlist):
    """
Returns the 'middle' score of the passed list.  If there is an even
number of scores, the mean of the 2 middle scores is returned.

Usage:   lmedianscore(inlist)
"""

    newlist = copy.deepcopy(inlist)
    newlist.sort()
    if len(newlist) % 2 == 0:   # if even number of scores, average middle 2
        index = len(newlist) / 2  # integer division correct
        median = float(newlist[index] + newlist[index - 1]) / 2
    else:
        index = len(newlist) / 2  # int divsion gives mid value when count from 0
        median = newlist[index]
    return median


def mode(inlist):
    """
Returns a list of the modal (most common) score(s) in the passed
list.  If there is more than one such score, all are returned.  The
bin-count for the mode(s) is also returned.

Usage:   lmode(inlist)
Returns: bin-count for mode(s), a list of modal value(s)
"""

    scores = pstat.unique(inlist)
    scores.sort()
    freq = []
    for item in scores:
        freq.append(inlist.count(item))
    maxfreq = max(freq)
    mode = []
    stillmore = 1
    while stillmore:
        try:
            indx = freq.index(maxfreq)
            mode.append(scores[indx])
            del freq[indx]
            del scores[indx]
        except ValueError:
            stillmore = 0
    return maxfreq, mode


####################################
############  MOMENTS  #############
####################################

def moment(inlist, moment=1):
    """
Calculates the nth moment about the mean for a sample (defaults to
the 1st moment).  Used to calculate coefficients of skewness and kurtosis.

Usage:   lmoment(inlist,moment=1)
Returns: appropriate moment (r) from pyLibrary. 1/n * SUM((inlist(i)-mean)**r)
"""
    if moment == 1:
        return 0.0
    else:
        mn = mean(inlist)
        n = len(inlist)
        s = 0
        for x in inlist:
            s = s + (x - mn) ** moment
        return s / float(n)


def variation(inlist):
    """
Returns the coefficient of variation, as defined in CRC Standard
Probability and Statistics, p.6.

Usage:   lvariation(inlist)
"""
    return 100.0 * samplestdev(inlist) / float(mean(inlist))


def skew(inlist):
    """
Returns the skewness of a distribution, as defined in Numerical
Recipies (alternate defn in CRC Standard Probability and Statistics, p.6.)

Usage:   lskew(inlist)
"""
    return moment(inlist, 3) / pow(moment(inlist, 2), 1.5)


def kurtosis(inlist):
    """
Returns the kurtosis of a distribution, as defined in Numerical
Recipies (alternate defn in CRC Standard Probability and Statistics, p.6.)

Usage:   lkurtosis(inlist)
"""
    return moment(inlist, 4) / pow(moment(inlist, 2), 2.0)


def describe(inlist):
    """
Returns some descriptive statistics of the passed list (assumed to be 1D).

Usage:   ldescribe(inlist)
Returns: n, mean, standard deviation, skew, kurtosis
"""
    n = len(inlist)
    mm = (min(inlist), max(inlist))
    m = mean(inlist)
    sd = stdev(inlist)
    sk = skew(inlist)
    kurt = kurtosis(inlist)
    return n, mm, m, sd, sk, kurt


####################################
#######  FREQUENCY STATS  ##########
####################################

def itemfreq(inlist):
    """
Returns a list of pairs.  Each pair consists of one of the scores in inlist
and it's frequency count.  Assumes a 1D list is passed.

Usage:   litemfreq(inlist)
Returns: a 2D frequency table (col [0:n-1]=scores, col n=frequencies)
"""
    scores = pstat.unique(inlist)
    scores.sort()
    freq = []
    for item in scores:
        freq.append(inlist.count(item))
    return zip(scores, freq)


def scoreatpercentile(inlist, percent):
    """
Returns the score at a given percentile relative to the distribution
given by inlist.

Usage:   lscoreatpercentile(inlist,percent)
"""
    if percent > 1:
        print "\nDividing percent>1 by 100 in lscoreatpercentile().\n"
        percent = percent / 100.0
    targetcf = percent * len(inlist)
    h, lrl, binsize, extras = histogram(inlist)
    cumhist = cumsum(copy.deepcopy(h))
    for i in range(len(cumhist)):
        if cumhist[i] >= targetcf:
            break
    score = binsize * ((targetcf - cumhist[i - 1]) / float(h[i])) + (lrl + binsize * i)
    return score


def percentileofscore(inlist, score, histbins=10, defaultlimits=None):
    """
Returns the percentile value of a score relative to the distribution
given by inlist.  Formula depends on the values used to histogram the data(!).

Usage:   lpercentileofscore(inlist,score,histbins=10,defaultlimits=None)
"""

    h, lrl, binsize, extras = histogram(inlist, histbins, defaultlimits)
    cumhist = cumsum(copy.deepcopy(h))
    i = int((score - lrl) / float(binsize))
    pct = (cumhist[i - 1] + ((score - (lrl + binsize * i)) / float(binsize)) * h[i]) / float(len(inlist)) * 100
    return pct


def histogram(inlist, numbins=10, defaultreallimits=None, printextras=0):
    """
Returns (i) a list of histogram bin counts, (ii) the smallest value
of the histogram binning, and (iii) the bin width (the last 2 are not
necessarily integers).  Default number of bins is 10.  If no sequence object
is given for defaultreallimits, the routine picks (usually non-pretty) bins
spanning all the numbers in the inlist.

Usage:   lhistogram (inlist, numbins=10, defaultreallimits=None,suppressoutput=0)
Returns: list of bin values, lowerreallimit, binsize, extrapoints
"""
    if (defaultreallimits <> None):
        if type(defaultreallimits) not in [ListType, TupleType] or len(defaultreallimits) == 1: # only one limit given, assumed to be lower one & upper is calc'd
            lowerreallimit = defaultreallimits
            upperreallimit = 1.000001 * max(inlist)
        else: # assume both limits given
            lowerreallimit = defaultreallimits[0]
            upperreallimit = defaultreallimits[1]
        binsize = (upperreallimit - lowerreallimit) / float(numbins)
    else:     # no limits given for histogram, both must be calc'd
        estbinwidth = (max(inlist) - min(inlist)) / float(numbins) + 1e-6 # 1=>cover all
        binsize = ((max(inlist) - min(inlist) + estbinwidth)) / float(numbins)
        lowerreallimit = min(inlist) - binsize / 2 # lower real limit,1st bin
    bins = [0] * (numbins)
    extrapoints = 0
    for num in inlist:
        try:
            if (num - lowerreallimit) < 0:
                extrapoints = extrapoints + 1
            else:
                bintoincrement = int((num - lowerreallimit) / float(binsize))
                bins[bintoincrement] = bins[bintoincrement] + 1
        except:
            extrapoints = extrapoints + 1
    if (extrapoints > 0 and printextras == 1):
        print '\nPoints outside given histogram range =', extrapoints
    return (bins, lowerreallimit, binsize, extrapoints)


def cumfreq(inlist, numbins=10, defaultreallimits=None):
    """
Returns a cumulative frequency histogram, using the histogram function.

Usage:   lcumfreq(inlist,numbins=10,defaultreallimits=None)
Returns: list of cumfreq bin values, lowerreallimit, binsize, extrapoints
"""
    h, l, b, e = histogram(inlist, numbins, defaultreallimits)
    cumhist = cumsum(copy.deepcopy(h))
    return cumhist, l, b, e


def relfreq(inlist, numbins=10, defaultreallimits=None):
    """
Returns a relative frequency histogram, using the histogram function.

Usage:   lrelfreq(inlist,numbins=10,defaultreallimits=None)
Returns: list of cumfreq bin values, lowerreallimit, binsize, extrapoints
"""
    h, l, b, e = histogram(inlist, numbins, defaultreallimits)
    for i in range(len(h)):
        h[i] = h[i] / float(len(inlist))
    return h, l, b, e


####################################
#####  VARIABILITY FUNCTIONS  ######
####################################

def obrientransform(*args):
    """
Computes a transform on input data (any number of columns).  Used to
test for homogeneity of variance prior to running one-way stats.  From
Maxwell and Delaney, p.112.

Usage:   lobrientransform(*args)
Returns: transformed data for use in an ANOVA
"""
    TINY = 1e-10
    k = len(args)
    n = [0.0] * k
    v = [0.0] * k
    m = [0.0] * k
    nargs = []
    for i in range(k):
        nargs.append(copy.deepcopy(args[i]))
        n[i] = float(len(nargs[i]))
        v[i] = var(nargs[i])
        m[i] = mean(nargs[i])
    for j in range(k):
        for i in range(n[j]):
            t1 = (n[j] - 1.5) * n[j] * (nargs[j][i] - m[j]) ** 2
            t2 = 0.5 * v[j] * (n[j] - 1.0)
            t3 = (n[j] - 1.0) * (n[j] - 2.0)
            nargs[j][i] = (t1 - t2) / float(t3)
    check = 1
    for j in range(k):
        if v[j] - mean(nargs[j]) > TINY:
            check = 0
    if check <> 1:
        raise ValueError, 'Problem in obrientransform.'
    else:
        return nargs


def samplevar(inlist):
    """
Returns the variance of the values in the passed list using
N for the denominator (i.e., DESCRIBES the sample variance only).

Usage:   lsamplevar(inlist)
"""
    n = len(inlist)
    mn = mean(inlist)
    deviations = []
    for item in inlist:
        deviations.append(item - mn)
    return ss(deviations) / float(n)


def samplestdev(inlist):
    """
Returns the standard deviation of the values in the passed list using
N for the denominator (i.e., DESCRIBES the sample stdev only).

Usage:   lsamplestdev(inlist)
"""
    return math.sqrt(samplevar(inlist))


def cov(x, y, keepdims=0):
    """
Returns the estimated covariance of the values in the passed
array (i.e., N-1).  Dimension can equal None (ravel array first), an
integer (the dimension over which to operate), or a sequence (operate
over multiple dimensions).  Set keepdims=1 to return an array with the
same number of dimensions as inarray.

Usage:   lcov(x,y,keepdims=0)
"""

    n = len(x)
    xmn = mean(x)
    ymn = mean(y)
    xdeviations = [0] * len(x)
    ydeviations = [0] * len(y)
    for i in range(len(x)):
        xdeviations[i] = x[i] - xmn
        ydeviations[i] = y[i] - ymn
    ss = 0.0
    for i in range(len(xdeviations)):
        ss = ss + xdeviations[i] * ydeviations[i]
    return ss / float(n - 1)


def var(inlist):
    """
Returns the variance of the values in the passed list using N-1
for the denominator (i.e., for estimating population variance).

Usage:   lvar(inlist)
"""
    n = len(inlist)
    mn = mean(inlist)
    deviations = [0] * len(inlist)
    for i in range(len(inlist)):
        deviations[i] = inlist[i] - mn
    return ss(deviations) / float(n - 1)


def stdev(inlist):
    """
Returns the standard deviation of the values in the passed list
using N-1 in the denominator (i.e., to estimate population stdev).

Usage:   lstdev(inlist)
"""
    return math.sqrt(var(inlist))


def sterr(inlist):
    """
Returns the standard error of the values in the passed list using N-1
in the denominator (i.e., to estimate population standard error).

Usage:   lsterr(inlist)
"""
    return stdev(inlist) / float(math.sqrt(len(inlist)))


def sem(inlist):
    """
Returns the estimated standard error of the mean (sx-bar) of the
values in the passed list.  sem = stdev / sqrt(n)

Usage:   lsem(inlist)
"""
    sd = stdev(inlist)
    n = len(inlist)
    return sd / math.sqrt(n)


def z(inlist, score):
    """
Returns the z-score for a given input score, given that score and the
list from which that score came.  Not appropriate for population calculations.

Usage:   lz(inlist, score)
"""
    z = (score - mean(inlist)) / samplestdev(inlist)
    return z


def zs(inlist):
    """
Returns a list of z-scores, one for each score in the passed list.

Usage:   lzs(inlist)
"""
    zscores = []
    for item in inlist:
        zscores.append(z(inlist, item))
    return zscores


####################################
#######  TRIMMING FUNCTIONS  #######
####################################

def trimboth(l, proportiontocut):
    """
Slices off the passed proportion of items from BOTH ends of the passed
list (i.e., with proportiontocut=0.1, slices 'leftmost' 10% AND 'rightmost'
10% of scores.  Assumes list is sorted by magnitude.  Slices off LESS if
proportion results in a non-integer slice index (i.e., conservatively
slices off proportiontocut).

Usage:   ltrimboth (l,proportiontocut)
Returns: trimmed version of list l
"""
    lowercut = int(proportiontocut * len(l))
    uppercut = len(l) - lowercut
    return l[lowercut:uppercut]


def trim1(l, proportiontocut, tail='right'):
    """
Slices off the passed proportion of items from ONE end of the passed
list (i.e., if proportiontocut=0.1, slices off 'leftmost' or 'rightmost'
10% of scores).  Slices off LESS if proportion results in a non-integer
slice index (i.e., conservatively slices off proportiontocut).

Usage:   ltrim1 (l,proportiontocut,tail='right')  or set tail='left'
Returns: trimmed version of list l
"""
    if tail == 'right':
        lowercut = 0
        uppercut = len(l) - int(proportiontocut * len(l))
    elif tail == 'left':
        lowercut = int(proportiontocut * len(l))
        uppercut = len(l)
    return l[lowercut:uppercut]


####################################
#####  CORRELATION FUNCTIONS  ######
####################################

def paired(x, y):
    """
Interactively determines the type of data and then runs the
appropriated statistic for paired group data.

Usage:   lpaired(x,y)
Returns: appropriate statistic name, value, and probability
"""
    samples = ''
    while samples not in ['i', 'r', 'I', 'R', 'c', 'C']:
        print '\nIndependent or related samples, or correlation (i,r,c): ',
        samples = raw_input()

    if samples in ['i', 'I', 'r', 'R']:
        print '\nComparing variances ...',
        # USE O'BRIEN'S TEST FOR HOMOGENEITY OF VARIANCE, Maxwell & delaney, p.112
        r = obrientransform(x, y)
        f, p = F_oneway(pstat.colex(r, 0), pstat.colex(r, 1))
        if p < 0.05:
            vartype = 'unequal, p=' + str(round(p, 4))
        else:
            vartype = 'equal'
        print vartype
        if samples in ['i', 'I']:
            if vartype[0] == 'e':
                t, p = ttest_ind(x, y, 0)
                print '\nIndependent samples t-test:  ', round(t, 4), round(p, 4)
            else:
                if len(x) > 20 or len(y) > 20:
                    z, p = ranksums(x, y)
                    print '\nRank Sums test (NONparametric, n>20):  ', round(z, 4), round(p, 4)
                else:
                    u, p = mannwhitneyu(x, y)
                    print '\nMann-Whitney U-test (NONparametric, ns<20):  ', round(u, 4), round(p, 4)

        else:  # RELATED SAMPLES
            if vartype[0] == 'e':
                t, p = ttest_rel(x, y, 0)
                print '\nRelated samples t-test:  ', round(t, 4), round(p, 4)
            else:
                t, p = ranksums(x, y)
                print '\nWilcoxon T-test (NONparametric):  ', round(t, 4), round(p, 4)
    else:  # CORRELATION ANALYSIS
        corrtype = ''
        while corrtype not in ['c', 'C', 'r', 'R', 'd', 'D']:
            print '\nIs the data Continuous, Ranked, or Dichotomous (c,r,d): ',
            corrtype = raw_input()
        if corrtype in ['c', 'C']:
            m, b, r, p, see = linregress(x, y)
            print '\nLinear regression for continuous variables ...'
            lol = [['Slope', 'Intercept', 'r', 'Prob', 'SEestimate'], [round(m, 4), round(b, 4), round(r, 4), round(p, 4), round(see, 4)]]
            pstat.printcc(lol)
        elif corrtype in ['r', 'R']:
            r, p = spearmanr(x, y)
            print '\nCorrelation for ranked variables ...'
            print "Spearman's r: ", round(r, 4), round(p, 4)
        else: # DICHOTOMOUS
            r, p = pointbiserialr(x, y)
            print '\nAssuming x contains a dichotomous variable ...'
            print 'Point Biserial r: ', round(r, 4), round(p, 4)
    print '\n\n'
    return None


def pearsonr(x, y):
    """
Calculates a Pearson correlation coefficient and the associated
probability value.  Taken from Heiman's Basic Statistics for the Behav.
Sci (2nd), p.195.

Usage:   lpearsonr(x,y)      where x and y are equal-length lists
Returns: Pearson's r value, two-tailed p-value
"""
    TINY = 1.0e-30
    if len(x) <> len(y):
        raise ValueError, 'Input values not paired in pearsonr.  Aborting.'
    n = len(x)
    x = map(float, x)
    y = map(float, y)
    xmean = mean(x)
    ymean = mean(y)
    r_num = n * (summult(x, y)) - sum(x) * sum(y)
    r_den = math.sqrt((n * ss(x) - square_of_sums(x)) * (n * ss(y) - square_of_sums(y)))
    r = (r_num / r_den)  # denominator already a float
    df = n - 2
    t = r * math.sqrt(df / ((1.0 - r + TINY) * (1.0 + r + TINY)))
    prob = betai(0.5 * df, 0.5, df / float(df + t * t))
    return r, prob


def lincc(x, y):
    """
Calculates Lin's concordance correlation coefficient.

Usage:   alincc(x,y)    where x, y are equal-length arrays
Returns: Lin's CC
"""
    covar = cov(x, y) * (len(x) - 1) / float(len(x))  # correct denom to n
    xvar = var(x) * (len(x) - 1) / float(len(x))  # correct denom to n
    yvar = var(y) * (len(y) - 1) / float(len(y))  # correct denom to n
    lincc = (2 * covar) / ((xvar + yvar) + ((mean(x) - mean(y)) ** 2))
    return lincc


def spearmanr(x, y):
    """
Calculates a Spearman rank-order correlation coefficient.  Taken
from Heiman's Basic Statistics for the Behav. Sci (1st), p.192.

Usage:   lspearmanr(x,y)      where x and y are equal-length lists
Returns: Spearman's r, two-tailed p-value
"""
    TINY = 1e-30
    if len(x) <> len(y):
        raise ValueError, 'Input values not paired in spearmanr.  Aborting.'
    n = len(x)
    rankx = rankdata(x)
    ranky = rankdata(y)
    dsq = sumdiffsquared(rankx, ranky)
    rs = 1 - 6 * dsq / float(n * (n ** 2 - 1))
    t = rs * math.sqrt((n - 2) / ((rs + 1.0) * (1.0 - rs)))
    df = n - 2
    probrs = betai(0.5 * df, 0.5, df / (df + t * t))  # t already a float
    # probability values for rs are from part 2 of the spearman function in
    # Numerical Recipies, p.510.  They are close to tables, but not exact. (?)
    return rs, probrs


def pointbiserialr(cats, vals):
    """
Calculates a point-biserial correlation coefficient and the associated
probability value.  Taken from Heiman's Basic Statistics for the Behav.
Sci (1st), p.194.

Usage:   pointbiserialr(x,y)      where x,y are equal-length lists
Returns: Point-biserial r, two-tailed p-value
"""
    TINY = 1e-30
    if len(cats) <> len(vals):
        raise ValueError, 'INPUT VALUES NOT PAIRED IN pointbiserialr.  ABORTING.'
    data = zip(cats, vals)
    categories = pstat.unique(cats)
    if len(categories) <> 2:
        raise ValueError, "Exactly 2 categories required for pointbiserialr()."
    else:   # there are 2 categories, continue
        c1 = [v for i, v in enumerate(vals) if cats[i] == categories[0]]
        c2 = [v for i, v in enumerate(vals) if cats[i] == categories[1]]
        xmean = mean(c1)
        ymean = mean(c2)
        n = len(vals)
        adjust = math.sqrt((len(c1) / float(n)) * (len(c2) / float(n)))
        rpb = (ymean - xmean) / samplestdev(vals) * adjust
        df = n - 2
        t = rpb * math.sqrt(df / ((1.0 - rpb + TINY) * (1.0 + rpb + TINY)))
        prob = betai(0.5 * df, 0.5, df / (df + t * t))  # t already a float
        return rpb, prob


def kendalltau(x, y):
    """
Calculates Kendall's tau ... correlation of ordinal data.  Adapted
from function kendl1 in Numerical Recipies.  Needs good test-routine.@@@

Usage:   lkendalltau(x,y)
Returns: Kendall's tau, two-tailed p-value
"""
    n1 = 0
    n2 = 0
    iss = 0
    for j in range(len(x) - 1):
        for k in range(j, len(y)):
            a1 = x[j] - x[k]
            a2 = y[j] - y[k]
            aa = a1 * a2
            if (aa):             # neither list has a tie
                n1 = n1 + 1
                n2 = n2 + 1
                if aa > 0:
                    iss = iss + 1
                else:
                    iss = iss - 1
            else:
                if (a1):
                    n1 = n1 + 1
                else:
                    n2 = n2 + 1
    tau = iss / math.sqrt(n1 * n2)
    svar = (4.0 * len(x) + 10.0) / (9.0 * len(x) * (len(x) - 1))
    z = tau / math.sqrt(svar)
    prob = erfcc(abs(z) / 1.4142136)
    return tau, prob


def linregress(x, y):
    """
Calculates a regression line on x,y pairs.

Usage:   llinregress(x,y)      x,y are equal-length lists of x-y coordinates
Returns: slope, intercept, r, two-tailed prob, sterr-of-estimate
"""
    TINY = 1.0e-20
    if len(x) <> len(y):
        raise ValueError, 'Input values not paired in linregress.  Aborting.'
    n = len(x)
    x = map(float, x)
    y = map(float, y)
    xmean = mean(x)
    ymean = mean(y)
    r_num = float(n * (summult(x, y)) - sum(x) * sum(y))
    r_den = math.sqrt((n * ss(x) - square_of_sums(x)) * (n * ss(y) - square_of_sums(y)))
    r = r_num / r_den
    z = 0.5 * math.log((1.0 + r + TINY) / (1.0 - r + TINY))
    df = n - 2
    t = r * math.sqrt(df / ((1.0 - r + TINY) * (1.0 + r + TINY)))
    prob = betai(0.5 * df, 0.5, df / (df + t * t))
    slope = r_num / float(n * ss(x) - square_of_sums(x))
    intercept = ymean - slope * xmean
    sterrest = math.sqrt(1 - r * r) * samplestdev(y)
    return slope, intercept, r, prob, sterrest


####################################
#####  INFERENTIAL STATISTICS  #####
####################################

def ttest_1samp(a, popmean):
    """
Calculates the t-obtained for the independent samples T-test on ONE group
of scores a, given a population mean.  Returns t-value, and prob.

Usage:   lttest_1samp(a,popmean)
Returns: t-value, two-tailed prob
"""
    x = mean(a)
    v = var(a)
    n = len(a)
    df = n - 1
    svar = ((n - 1) * v) / float(df)
    t = (x - popmean) / math.sqrt(svar * (1.0 / n))
    prob = betai(0.5 * df, 0.5, float(df) / (df + t * t))

    return t, prob


def ttest_ind(a, b):
    """
Calculates the t-obtained T-test on TWO INDEPENDENT samples of
scores a, and b.  From Numerical Recipies, p.483.  Returns t-value,
and prob.

Usage:   lttest_ind(a,b)
Returns: t-value, two-tailed prob
"""
    x1 = mean(a)
    x2 = mean(b)
    v1 = stdev(a) ** 2
    v2 = stdev(b) ** 2
    n1 = len(a)
    n2 = len(b)
    df = n1 + n2 - 2
    svar = ((n1 - 1) * v1 + (n2 - 1) * v2) / float(df)
    t = (x1 - x2) / math.sqrt(svar * (1.0 / n1 + 1.0 / n2))
    prob = betai(0.5 * df, 0.5, df / (df + t * t))

    return t, prob


def ttest_rel(a, b):
    """
Calculates the t-obtained T-test on TWO RELATED samples of scores,
a and b.  From Numerical Recipies, p.483.  Returns t-value,
and prob.

Usage:   lttest_rel(a,b)
Returns: t-value, two-tailed prob
"""
    if len(a) <> len(b):
        raise ValueError, 'Unequal length lists in ttest_rel.'
    x1 = mean(a)
    x2 = mean(b)
    v1 = var(a)
    v2 = var(b)
    n = len(a)
    cov = 0
    for i in range(len(a)):
        cov = cov + (a[i] - x1) * (b[i] - x2)
    df = n - 1
    cov = cov / float(df)
    sd = math.sqrt((v1 + v2 - 2.0 * cov) / float(n))
    t = (x1 - x2) / sd
    prob = betai(0.5 * df, 0.5, df / (df + t * t))

    return t, prob


def chisquare(f_obs, f_exp=None):
    """
Calculates a one-way chi square for list of observed frequencies and returns
the result.  If no expected frequencies are given, the total N is assumed to
be equally distributed across all groups.

Usage:   lchisquare(f_obs, f_exp=None)   f_obs = list of observed cell freq.
Returns: chisquare-statistic, associated p-value
"""
    k = len(f_obs)                 # number of groups
    if f_exp == None:
        f_exp = [sum(f_obs) / float(k)] * len(f_obs) # create k bins with = freq.
    chisq = 0
    for i in range(len(f_obs)):
        o = f_obs[i]
        e = f_exp[i]
        chisq = chisq + (o - e) ** 2 / float(e)
    return chisq, chisqprob(chisq, k - 1)


def ks_2samp(data1, data2):
    """
Computes the Kolmogorov-Smirnof statistic on 2 samples.  From
Numerical Recipies in C, page 493.

Usage:   lks_2samp(data1,data2)   data1&2 are lists of values for 2 conditions
Returns: KS D-value, associated p-value
"""
    j1 = 0
    j2 = 0
    fn1 = 0.0
    fn2 = 0.0
    n1 = len(data1)
    n2 = len(data2)
    en1 = n1
    en2 = n2
    d = 0.0
    data1.sort()
    data2.sort()
    while j1 < n1 and j2 < n2:
        d1 = data1[j1]
        d2 = data2[j2]
        if d1 <= d2:
            fn1 = (j1) / float(en1)
            j1 = j1 + 1
        if d2 <= d1:
            fn2 = (j2) / float(en2)
            j2 = j2 + 1
        dt = (fn2 - fn1)
        if math.fabs(dt) > math.fabs(d):
            d = dt
    try:
        en = math.sqrt(en1 * en2 / float(en1 + en2))
        prob = ksprob((en + 0.12 + 0.11 / en) * abs(d))
    except:
        prob = 1.0
    return d, prob


def mannwhitneyu(x, y):
    """
Calculates a Mann-Whitney U statistic on the provided scores and
returns the result.  Use only when the n in each condition is < 20 and
you have 2 independent samples of ranks.  NOTE: Mann-Whitney U is
significant if the u-obtained is LESS THAN or equal to the critical
value of U found in the tables.  Equivalent to Kruskal-Wallis H with
just 2 groups.

Usage:   lmannwhitneyu(data)
Returns: u-statistic, one-tailed p-value (i.e., p(z(U)))
"""
    n1 = len(x)
    n2 = len(y)
    ranked = rankdata(x + y)
    rankx = ranked[0:n1]       # get the x-ranks
    ranky = ranked[n1:]        # the rest are y-ranks
    u1 = n1 * n2 + (n1 * (n1 + 1)) / 2.0 - sum(rankx)  # calc U for x
    u2 = n1 * n2 - u1                            # remainder is U for y
    bigu = max(u1, u2)
    smallu = min(u1, u2)
    proportion = bigu / float(n1 * n2)
    T = math.sqrt(tiecorrect(ranked))  # correction factor for tied scores
    if T == 0:
        raise ValueError, 'All numbers are identical in lmannwhitneyu'
    sd = math.sqrt(T * n1 * n2 * (n1 + n2 + 1) / 12.0)
    z = abs((bigu - n1 * n2 / 2.0) / sd)  # normal approximation for prob calc
    return smallu, 1.0 - zprob(z) #, proportion


def tiecorrect(rankvals):
    """
Corrects for ties in Mann Whitney U and Kruskal Wallis H tests.  See
Siegel, S. (1956) Nonparametric Statistics for the Behavioral Sciences.
New York: McGraw-Hill.  Code adapted from |Stat rankind.c code.

Usage:   ltiecorrect(rankvals)
Returns: T correction factor for U or H
"""
    sorted, posn = shellsort(rankvals)
    n = len(sorted)
    T = 0.0
    i = 0
    while (i < n - 1):
        if sorted[i] == sorted[i + 1]:
            nties = 1
            while (i < n - 1) and (sorted[i] == sorted[i + 1]):
                nties = nties + 1
                i = i + 1
            T = T + nties ** 3 - nties
        i = i + 1
    T = T / float(n ** 3 - n)
    return 1.0 - T


def ranksums(x, y):
    """
Calculates the rank sums statistic on the provided scores and
returns the result.  Use only when the n in each condition is > 20 and you
have 2 independent samples of ranks.

Usage:   lranksums(x,y)
Returns: a z-statistic, two-tailed p-value
"""
    n1 = len(x)
    n2 = len(y)
    alldata = x + y
    ranked = rankdata(alldata)
    x = ranked[:n1]
    y = ranked[n1:]
    s = sum(x)
    expected = n1 * (n1 + n2 + 1) / 2.0
    z = (s - expected) / math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    prob = 2 * (1.0 - zprob(abs(z)))
    return z, prob


def wilcoxont(x, y):
    """
Calculates the Wilcoxon T-test for related samples and returns the
result.  A non-parametric T-test.

Usage:   lwilcoxont(x,y)
Returns: a t-statistic, two-tail probability estimate
"""
    if len(x) <> len(y):
        raise ValueError, 'Unequal N in wilcoxont.  Aborting.'
    d = []
    for i in range(len(x)):
        diff = x[i] - y[i]
        if diff <> 0:
            d.append(diff)
    count = len(d)
    absd = map(abs, d)
    absranked = rankdata(absd)
    r_plus = 0.0
    r_minus = 0.0
    for i in range(len(absd)):
        if d[i] < 0:
            r_minus = r_minus + absranked[i]
        else:
            r_plus = r_plus + absranked[i]
    wt = min(r_plus, r_minus)
    mn = count * (count + 1) * 0.25
    se = math.sqrt(count * (count + 1) * (2.0 * count + 1.0) / 24.0)
    z = math.fabs(wt - mn) / se
    prob = 2 * (1.0 - zprob(abs(z)))
    return wt, prob


def kruskalwallish(*args):
    """
The Kruskal-Wallis H-test is a non-parametric ANOVA for 3 or more
groups, requiring at least 5 subjects in each group.  This function
calculates the Kruskal-Wallis H-test for 3 or more independent samples
and returns the result.

Usage:   lkruskalwallish(*args)
Returns: H-statistic (corrected for ties), associated p-value
"""
    args = list(args)
    n = [0] * len(args)
    all = []
    n = map(len, args)
    for i in range(len(args)):
        all = all + args[i]
    ranked = rankdata(all)
    T = tiecorrect(ranked)
    for i in range(len(args)):
        args[i] = ranked[0:n[i]]
        del ranked[0:n[i]]
    rsums = []
    for i in range(len(args)):
        rsums.append(sum(args[i]) ** 2)
        rsums[i] = rsums[i] / float(n[i])
    ssbn = sum(rsums)
    totaln = sum(n)
    h = 12.0 / (totaln * (totaln + 1)) * ssbn - 3 * (totaln + 1)
    df = len(args) - 1
    if T == 0:
        raise ValueError, 'All numbers are identical in lkruskalwallish'
    h = h / float(T)
    return h, chisqprob(h, df)


def friedmanchisquare(*args):
    """
Friedman Chi-Square is a non-parametric, one-way within-subjects
ANOVA.  This function calculates the Friedman Chi-square test for repeated
measures and returns the result, along with the associated probability
value.  It assumes 3 or more repeated measures.  Only 3 levels requires a
minimum of 10 subjects in the study.  Four levels requires 5 subjects per
level(??).

Usage:   lfriedmanchisquare(*args)
Returns: chi-square statistic, associated p-value
"""
    k = len(args)
    if k < 3:
        raise ValueError, 'Less than 3 levels.  Friedman test not appropriate.'
    n = len(args[0])
    data = apply(zip, tuple(args))
    for i in range(len(data)):
        data[i] = rankdata(data[i])
    ssbn = 0
    for i in range(k):
        ssbn = ssbn + sum(args[i]) ** 2
    chisq = 12.0 / (k * n * (k + 1)) * ssbn - 3 * n * (k + 1)
    return chisq, chisqprob(chisq, k - 1)


####################################
####  PROBABILITY CALCULATIONS  ####
####################################

def chisqprob(chisq, df):
    """
Returns the (1-tailed) probability value associated with the provided
chi-square value and df.  Adapted from chisq.c in Gary Perlman's |Stat.

Usage:   lchisqprob(chisq,df)
"""
    BIG = 20.0

    def ex(x):
        BIG = 20.0
        if x < -BIG:
            return 0.0
        else:
            return math.exp(x)

    if chisq <= 0 or df < 1:
        return 1.0
    a = 0.5 * chisq
    if df % 2 == 0:
        even = 1
    else:
        even = 0
    if df > 1:
        y = ex(-a)
    if even:
        s = y
    else:
        s = 2.0 * zprob(-math.sqrt(chisq))
    if (df > 2):
        chisq = 0.5 * (df - 1.0)
        if even:
            z = 1.0
        else:
            z = 0.5
        if a > BIG:
            if even:
                e = 0.0
            else:
                e = math.log(math.sqrt(math.pi))
            c = math.log(a)
            while (z <= chisq):
                e = math.log(z) + e
                s = s + ex(c * z - a - e)
                z = z + 1.0
            return s
        else:
            if even:
                e = 1.0
            else:
                e = 1.0 / math.sqrt(math.pi) / math.sqrt(a)
            c = 0.0
            while (z <= chisq):
                e = e * (a / float(z))
                c = c + e
                z = z + 1.0
            return (c * y + s)
    else:
        return s


def erfcc(x):
    """
Returns the complementary error function erfc(x) with fractional
error everywhere less than 1.2e-7.  Adapted from Numerical Recipies.

Usage:   lerfcc(x)
"""
    z = abs(x)
    t = 1.0 / (1.0 + 0.5 * z)
    ans = t * math.exp(
        -z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    if x >= 0:
        return ans
    else:
        return 2.0 - ans


def zprob(z):
    """
Returns the area under the normal curve 'to the left of' the given z value.
Thus,
    for z<0, zprob(z) = 1-tail probability
    for z>0, 1.0-zprob(z) = 1-tail probability
    for any z, 2.0*(1.0-zprob(abs(z))) = 2-tail probability
Adapted from z.c in Gary Perlman's |Stat.

Usage:   lzprob(z)
"""
    Z_MAX = 6.0    # maximum meaningful z-value
    if z == 0.0:
        x = 0.0
    else:
        y = 0.5 * math.fabs(z)
        if y >= (Z_MAX * 0.5):
            x = 1.0
        elif (y < 1.0):
            w = y * y
            x = ((((((((0.000124818987 * w
                        - 0.001075204047) * w + 0.005198775019) * w
                      - 0.019198292004) * w + 0.059054035642) * w
                    - 0.151968751364) * w + 0.319152932694) * w
                  - 0.531923007300) * w + 0.797884560593) * y * 2.0
        else:
            y = y - 2.0
            x = (((((((((((((-0.000045255659 * y
                             + 0.000152529290) * y - 0.000019538132) * y
                           - 0.000676904986) * y + 0.001390604284) * y
                         - 0.000794620820) * y - 0.002034254874) * y
                       + 0.006549791214) * y - 0.010557625006) * y
                     + 0.011630447319) * y - 0.009279453341) * y
                   + 0.005353579108) * y - 0.002141268741) * y
                 + 0.000535310849) * y + 0.999936657524
    if z > 0.0:
        prob = ((x + 1.0) * 0.5)
    else:
        prob = ((1.0 - x) * 0.5)
    return prob


def ksprob(alam):
    """
Computes a Kolmolgorov-Smirnov t-test significance level.  Adapted from
Numerical Recipies.

Usage:   lksprob(alam)
"""
    fac = 2.0
    sum = 0.0
    termbf = 0.0
    a2 = -2.0 * alam * alam
    for j in range(1, 201):
        term = fac * math.exp(a2 * j * j)
        sum = sum + term
        if math.fabs(term) <= (0.001 * termbf) or math.fabs(term) < (1.0e-8 * sum):
            return sum
        fac = -fac
        termbf = math.fabs(term)
    return 1.0             # Get here only if fails to converge; was 0.0!!


def fprob(dfnum, dfden, F):
    """
Returns the (1-tailed) significance level (p-value) of an F
statistic given the degrees of freedom for the numerator (dfR-dfF) and
the degrees of freedom for the denominator (dfF).

Usage:   lfprob(dfnum, dfden, F)   where usually dfnum=dfbn, dfden=dfwn
"""
    p = betai(0.5 * dfden, 0.5 * dfnum, dfden / float(dfden + dfnum * F))
    return p


def betacf(a, b, x):
    """
    This function evaluates the continued fraction form of the incomplete
    Beta function, betai.  (Adapted from: Numerical Recipies in C.)

    Usage:   lbetacf(a,b,x)
    """
    ITMAX = 200
    EPS = 3.0e-7

    bm = az = am = 1.0
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    bz = 1.0 - qab * x / qap
    for i in range(ITMAX + 1):
        em = float(i + 1)
        tem = em + em
        d = em * (b - em) * x / ((qam + tem) * (a + tem))
        ap = az + d * am
        bp = bz + d * bm
        d = -(a + em) * (qab + em) * x / ((qap + tem) * (a + tem))
        app = ap + d * az
        bpp = bp + d * bz
        aold = az
        am = ap / bpp
        bm = bp / bpp
        az = app / bpp
        bz = 1.0
        if (abs(az - aold) < (EPS * abs(az))):
            return az
    print 'a or b too big, or ITMAX too small in Betacf.'


def gammln(xx):
    """
    Returns the gamma function of xx.
        Gamma(z) = Integral(0,infinity) of t^(z-1)exp(-t) dt.
    (Adapted from: Numerical Recipies in C.)

    Usage:   lgammln(xx)
    """

    coeff = [76.18009173, -86.50532033, 24.01409822, -1.231739516, 0.120858003e-2, -0.536382e-5]
    x = xx - 1.0
    tmp = x + 5.5
    tmp = tmp - (x + 0.5) * math.log(tmp)
    ser = 1.0
    for j in range(len(coeff)):
        x = x + 1
        ser = ser + coeff[j] / x
    return -tmp + math.log(2.50662827465 * ser)


def betai(a, b, x):
    """
Returns the incomplete beta function:

    I-sub-x(a,b) = 1/B(a,b)*(Integral(0,x) of t^(a-1)(1-t)^(b-1) dt)

where a,b>0 and B(a,b) = G(a)*G(b)/(G(a+b)) where G(a) is the gamma
function of a.  The continued fraction formulation is implemented here,
using the betacf function.  (Adapted from: Numerical Recipies in C.)

Usage:   lbetai(a,b,x)
"""
    if (x < 0.0 or x > 1.0):
        raise ValueError, 'Bad x in lbetai'

    if (x == 0.0 or x == 1.0):
        bt = 0.0
    else:
        bt = math.exp(gammln(a + b) - gammln(a) - gammln(b) + a * math.log(x) + b * math.log(1.0 - x))

    if (x < (a + 1.0) / (a + b + 2.0)):
        return bt * betacf(a, b, x) / float(a)
    else:
        return 1.0 - bt * betacf(b, a, 1.0 - x) / float(b)


####################################
#######  ANOVA CALCULATIONS  #######
####################################

def F_oneway(*lists):
    """
Performs a 1-way ANOVA, returning an F-value and probability given
any number of groups.  From Heiman, pp.394-7.

Usage:   F_oneway(*lists)    where *lists is any number of lists, one per
                                  treatment group
Returns: F value, one-tailed p-value
"""
    a = len(lists)           # ANOVA on 'a' groups, each in it's own list
    means = [0] * a
    vars = [0] * a
    ns = [0] * a
    alldata = []
    tmp = lists
    means = map(mean, tmp)
    vars = map(var, tmp)
    ns = map(len, lists)
    for i in range(len(lists)):
        alldata = alldata + lists[i]
    bign = len(alldata)
    sstot = ss(alldata) - (square_of_sums(alldata) / float(bign))
    ssbn = 0
    for list in lists:
        ssbn = ssbn + square_of_sums(list) / float(len(list))
    ssbn = ssbn - (square_of_sums(alldata) / float(bign))
    sswn = sstot - ssbn
    dfbn = a - 1
    dfwn = bign - a
    msb = ssbn / float(dfbn)
    msw = sswn / float(dfwn)
    f = msb / msw
    prob = fprob(dfbn, dfwn, f)
    return f, prob


def F_value(ER, EF, dfnum, dfden):
    """
Returns an F-statistic given the following:
        ER  = error associated with the null hypothesis (the Restricted model)
        EF  = error associated with the alternate hypothesis (the Full model)
        dfR-dfF = degrees of freedom of the numerator
        dfF = degrees of freedom associated with the denominator/Full model

Usage:   lF_value(ER,EF,dfnum,dfden)
"""
    return ((ER - EF) / float(dfnum) / (EF / float(dfden)))



def incr(l, cap):        # to increment a list up to a max-list of 'cap'
    """
Simulate a counting system from an n-dimensional list.

Usage:   lincr(l,cap)   l=list to increment, cap=max values for each list pos'n
Returns: next set of values for list l, OR -1 (if overflow)
"""
    l[0] = l[0] + 1     # e.g., [0,0,0] --> [2,4,3] (=cap)
    for i in range(len(l)):
        if l[i] > cap[i] and i < len(l) - 1: # if carryover AND not done
            l[i] = 0
            l[i + 1] = l[i + 1] + 1
        elif l[i] > cap[i] and i == len(l) - 1: # overflow past last column, must be finished
            l = -1
    return l


def cumsum(inlist):
    """
Returns a list consisting of the cumulative sum of the items in the
passed list.

Usage:   lcumsum(inlist)
"""
    newlist = copy.deepcopy(inlist)
    for i in range(1, len(newlist)):
        newlist[i] = newlist[i] + newlist[i - 1]
    return newlist


def ss(inlist):
    """
Squares each value in the passed list, adds up these squares and
returns the result.

Usage:   lss(inlist)
"""
    ss = 0
    for item in inlist:
        ss = ss + item * item
    return ss


def summult(list1, list2):
    """
Multiplies elements in list1 and list2, element by element, and
returns the sum of all resulting multiplications.  Must provide equal
length lists.

Usage:   lsummult(list1,list2)
"""
    if len(list1) <> len(list2):
        raise ValueError, "Lists not equal length in summult."
    s = 0
    for item1, item2 in zip(list1, list2):
        s = s + item1 * item2
    return s


def sumdiffsquared(x, y):
    """
Takes pairwise differences of the values in lists x and y, squares
these differences, and returns the sum of these squares.

Usage:   lsumdiffsquared(x,y)
Returns: sum[(x[i]-y[i])**2]
"""
    sds = 0
    for i in range(len(x)):
        sds = sds + (x[i] - y[i]) ** 2
    return sds


def square_of_sums(inlist):
    """
Adds the values in the passed list, squares the sum, and returns
the result.

Usage:   lsquare_of_sums(inlist)
Returns: sum(inlist[i])**2
"""
    s = sum(inlist)
    return float(s) * s


def shellsort(inlist):
    """
Shellsort algorithm.  Sorts a 1D-list.

Usage:   lshellsort(inlist)
Returns: sorted-inlist, sorting-index-vector (for original list)
"""
    n = len(inlist)
    svec = copy.deepcopy(inlist)
    ivec = range(n)
    gap = n / 2   # integer division needed
    while gap > 0:
        for i in range(gap, n):
            for j in range(i - gap, -1, -gap):
                while j >= 0 and svec[j] > svec[j + gap]:
                    temp = svec[j]
                    svec[j] = svec[j + gap]
                    svec[j + gap] = temp
                    itemp = ivec[j]
                    ivec[j] = ivec[j + gap]
                    ivec[j + gap] = itemp
        gap = gap / 2  # integer division needed
    # svec is now sorted inlist, and ivec has the order svec[i] = vec[ivec[i]]
    return svec, ivec


def rankdata(inlist):
    """
Ranks the data in inlist, dealing with ties appropritely.  Assumes
a 1D inlist.  Adapted from Gary Perlman's |Stat ranksort.

Usage:   rankdata(inlist)
Returns: a list of length equal to inlist, containing rank scores
"""
    n = len(inlist)
    svec, ivec = shellsort(inlist)
    sumranks = 0
    dupcount = 0
    newlist = [0] * n
    for i in range(n):
        sumranks = sumranks + i
        dupcount = dupcount + 1
        if i == n - 1 or svec[i] <> svec[i + 1]:
            averank = sumranks / float(dupcount) + 1
            for j in range(i - dupcount + 1, i + 1):
                newlist[ivec[j]] = averank
            sumranks = 0
            dupcount = 0
    return newlist


def findwithin(data):
    """
Returns an integer representing a binary vector, where 1=within-
subject factor, 0=between.  Input equals the entire data 2D list (i.e.,
column 0=random factor, column -1=measured values (those two are skipped).
Note: input data is in |Stat format ... a list of lists ("2D list") with
one row per measured value, first column=subject identifier, last column=
score, one in-between column per factor (these columns contain level
designations on each factor).  See also stats.anova.__doc__.

Usage:   lfindwithin(data)     data in |Stat format
"""

    numfact = len(data[0]) - 1
    withinvec = 0
    for col in range(1, numfact):
        examplelevel = pstat.unique(pstat.colex(data, col))[0]
        rows = pstat.linexand(data, col, examplelevel)  # get 1 level of this factor
        factsubjs = pstat.unique(pstat.colex(rows, 0))
        allsubjs = pstat.unique(pstat.colex(data, 0))
        if len(factsubjs) == len(allsubjs):  # fewer Ss than scores on this factor?
            withinvec = withinvec + (1 << col)
    return withinvec


