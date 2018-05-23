'''One-off script (really just a log of the steps I executed by hand)
to convert the Excel file I received for effective areas into a format that is
easier to read by MARXS.
'''
import astropy.units as u
from astropy.table import Table

# Path to marxs-lynx git:
datapath = '/melkor/d1/guenther/projects/Lynx/marxs-lynx/marxslynx/data/'

# pasted from the xlsx table
dat = '''1	 127 	 231 	 344 	 336 	 335 	 334 	 336 	 335 	 330 	 309 	 253
2	 242 	 346 	 702 	 677 	 674 	 672 	 675 	 674 	 657 	 590 	 429
3	 357 	 461 	 1,064 	 1,012 	 1,007 	 1,002 	 1,008 	 1,004 	 966 	 826 	 532
4	 472 	 576 	 1,427 	 1,339 	 1,329 	 1,320 	 1,330 	 1,320 	 1,250 	 1,007 	 573
5	 587 	 691 	 1,785 	 1,652 	 1,637 	 1,622 	 1,635 	 1,615 	 1,498 	 1,116 	 560
6	 702 	 806 	 2,119 	 1,933 	 1,911 	 1,887 	 1,903 	 1,866 	 1,683 	 1,126 	 502
7	 817 	 921 	 2,363 	 2,125 	 2,095 	 2,062 	 2,077 	 2,015 	 1,748 	 994 	 406
8	 932 	 1,036 	 2,595 	 2,299 	 2,260 	 2,215 	 2,227 	 2,128 	 1,744 	 752 	 304
9	 1,047 	 1,151 	 2,858 	 2,495 	 2,443 	 2,382 	 2,388 	 2,233 	 1,676 	 434 	 213
10	 1,162 	 1,266 	 3,059 	 2,630 	 2,564 	 2,485 	 2,478 	 2,250 	 1,462 	 171 	 135
11	 1,277 	 1,381 	 3,190 	 2,700 	 2,620 	 2,520 	 2,497 	 2,176 	 1,100 	 59 	 79
12	 1,392 	 1,496 	 3,416 	 2,845 	 2,745 	 2,617 	 2,569 	 2,109 	 672 	 24 	 45
'''
dat = dat.replace(',', '')
t = Table.read(dat, format='ascii.no_header', names=['Metashell Serial Number', 'r_inner',
                                                     'r_outer', '0.2', '0.4', '0.6', '0.8', '1.0', '1.4', '1.8', '2.0', '2.2'])
t['r_inner'].unit = u.mm
t['r_outer'].unit = u.mm
t.meta['ORIGFILE'] = 'EffectiveAreas4XGS.xlsx'
t.meta['origin'] = '''-------- Forwarded Message --------
Subject: 	FW: request for XGS
Date: 	Fri, 9 Mar 2018 23:57:57 +0000
From: 	Gaskin, Jessica A. (MSFC-ST12) <jessica.gaskin@nasa.gov>
To: 	Bautz <mwb@space.mit.edu>, Heilmann <ralf@space.mit.edu>, rlm90@psu.edu <rlm90@psu.edu>


Mark, Ralf, and Randy,

The attached is the EA vs energy vs radii for the meta-shell configuration. Do you need more fidelity than this?

-Best,
Jessica
________________________________________
From: Zhang, William W., Dr {Will} (GSFC-6620)
Sent: Friday, March 09, 2018 4:05 PM
To: Gaskin, Jessica A. (MSFC-ST12)
Subject: Re: request for XGS

See attachment.  Let me know if you want additional or different
information.


============================================================================================================
William W. Zhang ||  Goddard Space Flight Center || Bldg 34 Rm S266 || Greenbelt, MD 20771 || (301) 286 6230
============================================================================================================'''
t.meta['author'] = 'Hans Moritz Guenther'

t.write(datapath + 'metashellgeom.dat', format='ascii.ecsv', overwrite=True)
