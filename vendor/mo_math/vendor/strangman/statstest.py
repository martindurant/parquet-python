
import stats, os, pstat
reload(stats)

try:
    import numpy as N
except ImportError:
    pass

l = range(1,21)
lf = range(1,21)
lf[2] = 3.0
a = N.array(l)
af = N.array(lf)
ll = [l]*5
aa = N.array(ll)

print '\nCENTRAL TENDENCY'
print 'geometricmean:',stats.geometricmean(l), stats.geometricmean(lf), stats.geometricmean(a), stats.geometricmean(af)
print 'harmonicmean:',stats.harmonicmean(l), stats.harmonicmean(lf), stats.harmonicmean(a), stats.harmonicmean(af)
print 'mean:',stats.mean(l), stats.mean(lf), stats.mean(a), stats.mean(af)
print 'median:',stats.median(l),stats.median(lf),stats.median(a),stats.median(af)
print 'medianscore:',stats.medianscore(l),stats.medianscore(lf),stats.medianscore(a),stats.medianscore(af)
print 'mode:',stats.mode(l),stats.mode(a)

print '\nMOMENTS'
print 'moment:',stats.moment(l),stats.moment(lf),stats.moment(a),stats.moment(af)
print 'variation:',stats.variation(l),stats.variation(a),stats.variation(lf),stats.variation(af)
print 'skew:',stats.skew(l),stats.skew(lf),stats.skew(a),stats.skew(af)
print 'kurtosis:',stats.kurtosis(l),stats.kurtosis(lf),stats.kurtosis(a),stats.kurtosis(af)
print 'mean:',stats.mean(a),stats.mean(af)
print 'var:',stats.var(a),stats.var(af)
print 'stdev:',stats.stdev(a),stats.stdev(af)
print 'sem:',stats.sem(a),stats.sem(af)
print 'describe:'
print stats.describe(l)
print stats.describe(lf)
print stats.describe(a)
print stats.describe(af)

print '\nFREQUENCY'
print 'freqtable:'
print 'itemfreq:'
print stats.itemfreq(l)
print stats.itemfreq(a)
print 'scoreatpercentile:',stats.scoreatpercentile(l,40),stats.scoreatpercentile(lf,40),stats.scoreatpercentile(a,40),stats.scoreatpercentile(af,40)
print 'percentileofscore:',stats.percentileofscore(l,12),stats.percentileofscore(lf,12),stats.percentileofscore(a,12),stats.percentileofscore(af,12)
print 'histogram:',stats.histogram(l),stats.histogram(a)
print 'cumfreq:'
print stats.cumfreq(l)
print stats.cumfreq(lf)
print stats.cumfreq(a)
print stats.cumfreq(af)
print 'relfreq:'
print stats.relfreq(l)
print stats.relfreq(lf)
print stats.relfreq(a)
print stats.relfreq(af)

print '\nVARIATION'
print 'obrientransform:'

l = range(1,21)
a = N.array(l)
ll = [l]*5
aa = N.array(ll)

print stats.obrientransform(l,l,l,l,l)
print stats.obrientransform(a,a,a,a,a)

print 'samplevar:',stats.samplevar(l),stats.samplevar(a)
print 'samplestdev:',stats.samplestdev(l),stats.samplestdev(a)
print 'var:',stats.var(l),stats.var(a)
print 'stdev:',stats.stdev(l),stats.stdev(a)
print 'sterr:',stats.sterr(l),stats.sterr(a)
print 'sem:',stats.sem(l),stats.sem(a)
print 'z:',stats.z(l,4),stats.z(a,4)
print 'zs:'
print stats.zs(l)
print stats.zs(a)

print '\nTRIMMING'
print 'trimboth:'
print stats.trimboth(l,.2)
print stats.trimboth(lf,.2)
print stats.trimboth(a,.2)
print stats.trimboth(af,.2)
print 'trim1:'
print stats.trim1(l,.2)
print stats.trim1(lf,.2)
print stats.trim1(a,.2)
print stats.trim1(af,.2)

print '\nCORRELATION'
# execfile('testpairedstats.py')

l = range(1,21)
a = N.array(l)
ll = [l]*5
aa = N.array(ll)

m = range(4,24)
m[10] = 34
b = N.array(m)

pb = [0]*9 + [1]*11
apb = N.array(pb)

print 'paired:'
# stats.paired(l,m)
# stats.paired(a,b)

print
print
print 'pearsonr:'
print stats.pearsonr(l,m)
print stats.pearsonr(a,b)
print 'spearmanr:'
print stats.spearmanr(l,m)
print stats.spearmanr(a,b)
print 'pointbiserialr:'
print stats.pointbiserialr(pb,l)
print stats.pointbiserialr(apb,a)
print 'kendalltau:'
print stats.kendalltau(l,m)
print stats.kendalltau(a,b)
print 'linregress:'
print stats.linregress(l,m)
print stats.linregress(a,b)

print '\nINFERENTIAL'
print 'ttest_1samp:'
print stats.ttest_1samp(l,12)
print stats.ttest_1samp(a,12)
print 'ttest_ind:'
print stats.ttest_ind(l,m)
print stats.ttest_ind(a,b)
print 'ttest_rel:'
print stats.ttest_rel(l,m)
print stats.ttest_rel(a,b)
print 'chisquare:'
print stats.chisquare(l)
print stats.chisquare(a)
print 'ks_2samp:'
print stats.ks_2samp(l,m)
print stats.ks_2samp(a,b)

print 'mannwhitneyu:'
print stats.mannwhitneyu(l,m)
print stats.mannwhitneyu(a,b)
print 'ranksums:'
print stats.ranksums(l,m)
print stats.ranksums(a,b)
print 'wilcoxont:'
print stats.wilcoxont(l,m)
print stats.wilcoxont(a,b)
print 'kruskalwallish:'
print stats.kruskalwallish(l,m,l)
print len(l), len(m)
print stats.kruskalwallish(a,b,a)
print 'friedmanchisquare:'
print stats.friedmanchisquare(l,m,l)
print stats.friedmanchisquare(a,b,a)

l = range(1,21)
a = N.array(l)
ll = [l]*5
aa = N.array(ll)

m = range(4,24)
m[10] = 34
b = N.array(m)

print '\n\nF_oneway:'
print stats.F_oneway(l,m)
print stats.F_oneway(a,b)
# print 'F_value:',stats.F_value(l),stats.F_value(a)

print '\nSUPPORT'
print 'sum:',stats.sum(l),stats.sum(lf),stats.sum(a),stats.sum(af)
print 'cumsum:'
print stats.cumsum(l)
print stats.cumsum(lf)
print stats.cumsum(a)
print stats.cumsum(af)
print 'ss:',stats.ss(l),stats.ss(lf),stats.ss(a),stats.ss(af)
print 'summult:',stats.summult(l,m),stats.summult(lf,m),stats.summult(a,b),stats.summult(af,b)
print 'sumsquared:',stats.square_of_sums(l),stats.square_of_sums(lf),stats.square_of_sums(a),stats.square_of_sums(af)
print 'sumdiffsquared:',stats.sumdiffsquared(l,m),stats.sumdiffsquared(lf,m),stats.sumdiffsquared(a,b),stats.sumdiffsquared(af,b)
print 'shellsort:'
print stats.shellsort(m)
print stats.shellsort(b)
print 'rankdata:'
print stats.rankdata(m)
print stats.rankdata(b)

print '\nANOVAs'
execfile('testanova.py')

