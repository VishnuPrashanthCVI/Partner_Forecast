# Partner_Forecast
Summary:  This python code forecasts sales and partner performance in a multi tier distribution channel.  The current development completes the filling of the spare data matrices using pylibfm, a matrix factorization library, identifies the important categorical data,e.g., level of training, in estimating sales, and random forests for the final estimation of sales by quarter.  The final stage achieved a remarkable 90% p value on the test data.

Sample:  The data has been derived from multiple sources and randomized.  It follows the essential structure of partner management data using historical sales, categorical events, vendor ratings, and dealer opinions of the vendor capabilities.  The data itself is a mixture of continuous, ordinal, and categorical data.  Manipulating the data is still a manual exercise.  Many functions for manipulating the data is included in the data.py library.

Ordinal Data:  The ordinal data results from rankings made by both the vendor and the partner network.  This data is typically very sparse and in this case is composed of nearly 58% empty data cells.  The matrix factorization library is effective given the high accuracy of the final analytical stage.  The train/test split recorded a p value of .825 and an RMSE of .59.  The typical ordinal analytics using a fill with means records an RMSE of 1.2 on the 0 to 5 Likert scale responses.  NOTE:  This is randomized data which implies real data may have a different 'fit'. 

			Typical Ordinal Data Query
Loyalty				Satisfaction 		Value				Ease
Relationship		Profit				Communications		Product_Quality
Support_Quality		Partnership_Commit	Support_Importance	Will_Recommend	
Continue			Sales_Expectation	Competitor_Comparison
Acct_Mgr_Quality	Support_Mgr_Quality	Team_Comparison		Acct_Team_Ability
Acct_Team_Problem_Solver				Acct_Team_Info		Acct_Team_Plan
Acct_Team_Contribution					Acct_Team_Sales_Support
Operation_Support_Quality				Acct_Team_Satisfaction
Acct_Team_Engagement	Sales_Training	Service_Account_Team_Satisfaction
Service_Support		Service_Sales_Support					Solutions_Support
Joint_Service_Delivery	Online_Tools/Applications_Satisfaction
Online_Tools_Ease	Online_Tools_Effectiveness	Online_Tools_Colloboration
Online_Tools_Timeliness					Online_Tools_Sufficient
Online_Tools_Help	Acct_Team_Leadership	Acct_Team_Breadth

These queries have been condensed to general categories but the intent of the query is easily interpreted.

Categorical Data:




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

![alt text](features.png "Relative Feature Importance")

