# Copyright (c) 1999-2007 Gary Strangman; All Rights Reserved.
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
pstat.py module

#################################################
#######  Written by:  Gary Strangman  ###########
#######  Last modified:  Dec 18, 2007 ###########
#################################################

This module provides some useful list and array manipulation routines
modeled after those found in the |Stat package by Gary Perlman, plus a
number of other useful list/file manipulation functions.  The list-based
functions include:

      colex (listoflists,cnums)
      collapse (listoflists,keepcols,collapsecols,fcn1=None,fcn2=None,cfcn=None)
      dm (listoflists,criterion)
      flat (l)
      linexand (listoflists,columnlist,valuelist)
      linexor (listoflists,columnlist,valuelist)
      linedelimited (inlist,delimiter)
      lineincols (inlist,colsize)
      lineincustcols (inlist,colsizes)
      list2string (inlist)
      makelol(inlist)
      makestr(x)
      printcc (lst,extra=2)
      printincols (listoflists,colsize)
      pl (listoflists)
      printl(listoflists)
      replace (lst,oldval,newval)
      recode (inlist,listmap,cols='all')
      remap (listoflists,criterion)
      roundlist (inlist,num_digits_to_round_floats_to)
      sortby(listoflists,sortcols)
      unique (inlist)
      duplicates(inlist)
      writedelimited (listoflists, delimiter, file, writetype='w')

Some of these functions have alternate versions which are defined only if
Numeric (NumPy) can be imported.  These functions are generally named as
above, with an 'a' prefix.

      aabut (source, *args)
      acolex (a,indices,axis=1)
      acollapse (a,keepcols,collapsecols,sterr=0,ns=0)
      adm (a,criterion)
      alinexand (a,columnlist,valuelist)
      alinexor (a,columnlist,valuelist)
      areplace (a,oldval,newval)
      arecode (a,listmap,col='all')
      arowcompare (row1, row2)
      arowsame (row1, row2)
      asortrows(a,axis=0)
      aunique(inarray)
      aduplicates(inarray)

Currently, the code is all but completely un-optimized.  In many cases, the
array versions of functions amount simply to aliases to built-in array
functions/methods.  Their inclusion here is for function name consistency.
"""

## CHANGE LOG:
## ==========
## 07-11-26 ... edited to work with numpy
## 01-11-15 ... changed list2string() to accept a delimiter
## 01-06-29 ... converted exec()'s to eval()'s to make compatible with Py2.1
## 01-05-31 ... added duplicates() and aduplicates() functions
## 00-12-28 ... license made GPL, docstring and import requirements
## 99-11-01 ... changed version to 0.3
## 99-08-30 ... removed get, getstrings, put, aget, aput (into io.py)
## 03/27/99 ... added areplace function, made replace fcn recursive
## 12/31/98 ... added writefc function for ouput to fixed column sizes
## 12/07/98 ... fixed import problem (failed on collapse() fcn)
##              added __version__ variable (now 0.2)
## 12/05/98 ... updated doc-strings
##              added features to collapse() function
##              added flat() function for lists
##              fixed a broken asortrows()
## 11/16/98 ... fixed minor bug in aput for 1D arrays
##
## 11/08/98 ... fixed aput to output large arrays correctly

from types import *

__version__ = 0.4

###===========================  LIST FUNCTIONS  ==========================
###
### Here are the list functions, DEFINED FOR ALL SYSTEMS.
### Array functions (for NumPy-enabled computers) appear below.
###





def colex(listoflists, cnums):
    """
Extracts from listoflists the columns specified in the list 'cnums'
(cnums can be an integer, a sequence of integers, or a string-expression that
corresponds to a slice operation on the variable x ... e.g., 'x[3:]' will colex
columns 3 onward from the listoflists).

Usage:   colex (listoflists,cnums)
Returns: a list-of-lists corresponding to the columns from listoflists
         specified by cnums, in the order the column numbers appear in cnums
"""
    global index
    column = 0
    if type(cnums) in [ListType, TupleType]:   # if multiple columns to get
        index = cnums[0]
        column = map(lambda x: x[index], listoflists)
        for col in cnums[1:]:
            index = col
            column = zip(column, map(lambda x: x[index], listoflists))
    elif type(cnums) == StringType:              # if an 'x[3:]' type expr.
        evalstring = 'map(lambda x: x' + cnums + ', listoflists)'
        column = eval(evalstring)
    else:                                     # else it's just 1 col to get
        index = cnums
        column = map(lambda x: x[index], listoflists)
    return column



def linexand(listoflists, columnlist, valuelist):
    """
Returns the rows of a list of lists where col (from columnlist) = val
(from valuelist) for EVERY pair of values (columnlist[i],valuelists[i]).
len(columnlist) must equal len(valuelist).

Usage:   linexand (listoflists,columnlist,valuelist)
Returns: the rows of listoflists where columnlist[i]=valuelist[i] for ALL i
"""
    if type(columnlist) not in [ListType, TupleType]:
        columnlist = [columnlist]
    if type(valuelist) not in [ListType, TupleType]:
        valuelist = [valuelist]
    criterion = ''
    for i in range(len(columnlist)):
        if type(valuelist[i]) == StringType:
            critval = '\'' + valuelist[i] + '\''
        else:
            critval = str(valuelist[i])
        criterion = criterion + ' x[' + str(columnlist[i]) + ']==' + critval + ' and'
    criterion = criterion[0:-3]         # remove the "and" after the last crit
    function = 'filter(lambda x: ' + criterion + ',listoflists)'
    lines = eval(function)
    return lines


def unique(inlist):
    """
Returns all unique items in the passed list.  If the a list-of-lists
is passed, unique LISTS are found (i.e., items in the first dimension are
compared).

Usage:   unique (inlist)
Returns: the unique elements (or rows) in inlist
"""
    uniques = []
    for item in inlist:
        if item not in uniques:
            uniques.append(item)
    return uniques

