# Partner_Forecast
Summary:  This python code forecasts sales and partner performance in a multi tier distribution channel.  The current development completes the filling of the spare data matrices using pylibfm, a matrix factorization library, identifies the important categorical data,e.g., level of training, in estimating sales, and random forests for the final estimation of sales by quarter.  The final stage achieved a remarkable 90% p value on the test data.
Next Development:  The next model to be added is the switchpoint time series analysis using PYMC.  The Probabilistic model will be applied to the revenue history for each partner demonstrating high or low performance in the predicted quarter.

Intent:  This description demonstrates the effectiveness of the program but does not intend to show the range of conclusions that are surfaced by the analysis.  For example, the causes of each partners performance is not displayed in this abbreviated description.

Program:  Assuming all dependencies have been loaded,e.g., pylibfm, the primary program is gazelle.py.  Open ipython and enter 'run gazelle.py'.  The charts require manual intervention.  This program still requires further streamlining.  The program data.py is a file of useful tools and scratch.py is just a space for scripting out solution prototypes.

Sample:  The data has been derived from multiple sources and randomized.  It follows the essential structure of partner management data using historical sales, categorical events, vendor ratings, and dealer opinions of the vendor capabilities.  The data itself is a mixture of continuous, ordinal, and categorical data.  Manipulating the data is still a manual exercise.  Many functions for manipulating the data are included in the data.py script library.

Ordinal Data:  The ordinal data results from rankings made by both the vendor and the partner network.  This data is typically very sparse and in this case is composed of nearly 58% empty data cells.  The matrix factorization library is effective given the high accuracy of the final analytical stage.  The train/test split recorded a p value of .825 and an RMSE of .59.  The typical ordinal analytics using a fill with means records an RMSE of 1.2 on the 0 to 5 Likert scale responses.  The RMSE is roughly halved using the matrix factorization routine.  NOTE:  This is randomized data which implies real data may have a different 'fit'. 

Typical Ordinal Data Query

Loyalty
Satisfaction
Value
Ease
Relationship
Profit
Communications
Product_Quality
Support_Quality
Partnership_Commit
Support_Importance
Will_Recommend
Continue
Sales_Expectation
Competitor_Comparison
Acct_Mgr_Quality
Support_Mgr_Quality	Team_Comparison
Acct_Team_Ability
Acct_Team_Problem_Solver
Acct_Team_Info		
Acct_Team_Plan
Acct_Team_Contribution
Acct_Team_Sales_Support
Operation_Support_Quality
Acct_Team_Satisfaction
Acct_Team_Engagement
Sales_Training
Service_Account_Team_Satisfaction
Service_Support
Service_Sales_Support
Solutions_Support
Joint_Service_Delivery
Online_Tools/Applications_Satisfaction
Online_Tools_Ease
Online_Tools_Effectiveness
Online_Tools_Colloboration
Online_Tools_Timeliness
Online_Tools_Sufficient
Online_Tools_Help
Acct_Team_Leadership
Acct_Team_Breadth

These queries have been abstracted/condensed to general categories but the intent of the query is easily interpreted.

Categorical Data:  Categorical data is separated from the other types and analyzed through a decision tree regressor after dummies are rendered.  After several iterations reducing the variables it was found that in this data the categorical information had little effect on the revenue prediction.  A p value of .15 resulted.  However, the technique is still in the ensemble string for future data set analytics.
	
		Typical Categorical Data
IDX	   							A unique index of all entries.
NID								A unique partner identifier.
Year							Data year.
Qtr								Data quarter in year.
Region							Major region or continent (often plural).
Country							Country of operation (often plural for an NID)
RTM								Route to market (e.g., VAR)
Certification					Vendor awarded certification.
Credit_Score					Vendor awarded credit limit.
Function						Occupation of reviewer, e.g., executive.
Customer_Size					Typical customer size by employees.
Customer_Segment				Typical customer, eg. GOVT, ENT.

Continuous Data:  The y dependent is the continuous variable revenue. Independent features are price discount and average invoice value.  Any number of additional features could be added.

Random Forests Regressor:  After distilling the data by the above steps the RFC was applied across the 5 quarters of data.  

THE RFC METRICS ARE AN RMSE OF 52.2940999046 (COMPARED TO RANGE OF 90,000 TO 135,000), P VALUE ON THE TRAINING SET OF .985 AND P VALUE ON THE TEST SET OF .90.
The predictions were separated into three groups: partners who exceeded historical performance by 10%, stayed the same within +-10%,declined by 10%.

The tabulation showed that partners who outperformed contributed an increase of $50M in revenue while underperformers reduced revenue to 69M.  Finding the cause of under performing partners would have a $6M to $7M positive impace on total revenue.  

				Rank		Player		Revenue
				Up			414			50,444
				Same		768			88,314
				Down		744			69,422

The causes of revenue are demonstrated in the analysis of features.	
	
![alt text](features.png "Relative Feature Importance")

The relative contribution by class is demonstrated:

![alt text](Partners.png "Partner Performance")


![alt text](Partners_Revenue.png "Partner Performance")