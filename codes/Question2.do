
*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??
*?? step 0. initial setup 
*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??

clear all
set more off

set maxvar 32767
set varabbrev off

if	"`c(username)'" == "wang" {
    global user "E:/RA/BoothTests"
}

cd "${user}"

global codes   "${user}/codes"
global data    "${user}/data"
global results "${user}/results"

version 17.0

*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??
*?? question 2.2. - Dataset Construction 
*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??

*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?
*-? question 2.2. - i.
*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?

import delimited "${data}/trade.csv", clear

collapse (sum) trade, by(origin destination year)
sort origin destination year 

*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?
*-? question 2.2. - ii.
*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?

preserve 
    import delimited "${data}/gravity.csv", clear
    save "${data}/temp_gravity.dta", replace 
restore 

merge 1:1 origin destination year using "${data}/temp_gravity.dta", keep(match) nogenerate

*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?
*-? question 2.2. - iii.
*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?

summarize trade if year==2015, detail
/* 
                         (sum) trade
-------------------------------------------------------------
      Percentiles      Smallest
 1%         .044           .001
 5%         .512           .001
10%        2.597           .001       Obs              28,803
25%       44.867           .001       Sum of wgt.      28,803

50%     1161.257                      Mean             311065
                        Largest       Std. dev.       3192502
75%     24663.26       1.05e+08
90%     234609.7       1.32e+08       Variance       1.02e+13
95%     772007.7       1.44e+08       Skewness       41.05018
99%      5641435       2.90e+08       Kurtosis       2798.787
*/

*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??
*?? question 2.3. - Estimation  
*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??*??

*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?
*-? question 2.3. - i.
*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?

generate log_distance = log(distance)
generate log_trade    = log(trade)

binscatter log_trade log_distance if year==2015 ///
    , nquantiles(30) reportreg ///
    xtitle("distance, in logs") ytitle("trade flow, in logs")
graph export "${results}/binscatter_trade_distance.png", replace as(png)

correlate log_trade log_distance if year==2015

*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?
*-? question 2.3. - ii.
*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?

global year_list 1995 2000 2005 2010 2015 2019 

matrix coeff_mat = J(6, 1, .)
matrix lb_mat    = J(6, 1, .)
matrix ub_mat    = J(6, 1, .)

egen double_cluster=group(origin destination)

local i = 1
foreach year in $year_list {
    regress log_trade log_distance if year==`year', vce(cluster double_cluster) 
        lincom log_distance
        matrix coeff_mat[`i',1] = r(estimate)
        matrix lb_mat[`i',1]    = r(lb)
        matrix ub_mat[`i',1]    = r(ub)
    local i = `i' + 1
}

local i = 1
capture drop year_index
generate year_index = .
foreach year in $year_list {
    replace year_index = `year' if _n==`i'
    local i = `i' + 1
}

matrix final_res = coeff_mat, lb_mat, ub_mat
matrix colnames final_res = coeff_Cluster lb_Cluster ub_Cluster
svmat  final_res, names(col)


twoway ///
    (rbar ub_Cluster lb_Cluster year_index, bcolor(ebblue) barwidth(0.3) vertical) ///
    (scatter coeff_Cluster year_index, lcolor(ebblue) mcolor(white) mfcolor(white) msymbol(D) msize(0.9)) ///
    , xlabel(1995 "1995" 2000 "2000" 2005 "2005" 2010 "2010" 2015 "2015" 2019 "2019") ///
    xtitle("Year") ytitle("Estimated coefficients") ///
    legend(off)
graph export "${results}/DistanceCoeffAcrossYears_Cluster.png", as(png) replace 


*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?
*-? question 2.3. - iii.
*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?

global year_list 1995 2000 2005 2010 2015 2019 

matrix coeff_mat = J(6, 1, .)
matrix lb_mat    = J(6, 1, .)
matrix ub_mat    = J(6, 1, .)

local i = 1
foreach year in $year_list {
    reghdfe log_trade log_distance if year==`year', vce(robust) absorb(origin#year destination#year) 
        lincom log_distance
        matrix coeff_mat[`i',1] = r(estimate)
        matrix lb_mat[`i',1]    = r(lb)
        matrix ub_mat[`i',1]    = r(ub)
    local i = `i' + 1
}

local i = 1
capture drop year_index
generate year_index = .
foreach year in $year_list {
    replace year_index = `year' if _n==`i'
    local i = `i' + 1
}

matrix final_res = coeff_mat, lb_mat, ub_mat
matrix colnames final_res = coeff_FE lb_FE ub_FE
svmat  final_res, names(col)


twoway ///
    (rbar ub_FE lb_FE year_index, bcolor(ebblue) barwidth(0.3) vertical) ///
    (scatter coeff_FE year_index, lcolor(ebblue) mcolor(white) mfcolor(white) msymbol(D) msize(0.9)) ///
    , xlabel(1995 "1995" 2000 "2000" 2005 "2005" 2010 "2010" 2015 "2015" 2019 "2019") ///
    xtitle("Year") ytitle("Estimated coefficients") ///
    legend(off)
graph export "${results}/DistanceCoeffAcrossYears_FE.png", as(png) replace 


*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?
*-? question 2.3. - iv.
*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?*-?

label variable log_distance "distance, in logs"
label variable contiguity   "sharing a border"
label variable language     "sharing a language"
label variable colonial     "having colonial ties"
label variable rta          "having a trade agreement"


reghdfe log_trade log_distance contiguity language colonial rta if year==2015, vce(robust) absorb(origin#year destination#year) 
    eststo model

esttab model using "${results}/regressiontable.tex", ///
    replace style(tex) fragment nocons label nofloat nobaselevels noobs ///
    nomtitles collabels(,none) ///
    star(* 0.10 ** 0.05 *** 0.01) ///
    keep(log_distance contiguity language colonial rta) ///
    order(log_distance contiguity language colonial rta) ///
    b(3) se(2) ///
    stats(r2 N, labels("R-squared" "Obs") fmt(%9.3f %9.0g)) ///
    prehead("\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}" "\begin{tabular}{lc}" "\toprule" "\toprule" "& \multicolumn{1}{c}{Trade flows} \\ ") ///
    posthead("\hline") ///
    prefoot("\hline") ///
    postfoot("\hline" "\hline" "\end{tabular}" "\begin{tablenotes}" "\footnotesize" "\item" "Notes. Sample includes trade flows between different countries in 2015. Importer and exporter fixed effects are controlled. Robust standard error are reported. " "\end{tablenotes}")
