
Create database Pancham_Silage_Factory_DB;
use Pancham_Silage_Factory_DB;



SET GLOBAL local_infile = 1;
SHOW VARIABLES LIKE 'local_infile';
SHOW VARIABLES LIKE 'secure_file_priv';



CREATE TABLE main_data (
    Transaction_ID VARCHAR(25) PRIMARY KEY,
    Date DATE,

    Customer_Type VARCHAR(50),
    Crop_Type VARCHAR(50),
    Harvest_Season VARCHAR(30),

    Moisture_Content_Percent DECIMAL(5,2),
    DM_Content_Percent DECIMAL(5,2),

    Quantity_MT DECIMAL(10,2),
    Price_per_MT_INR DECIMAL(12,2),
    Total_Sales_INR DECIMAL(14,2),

    Logistics_Cost_INR DECIMAL(14,2),
    Profit_Margin_Percent DECIMAL(5,2),

    Payment_Mode VARCHAR(30),
    Credit_Period_Days INT,
    Bagging_Type VARCHAR(30)
);




LOAD DATA LOCAL INFILE
"C:/kalpesh/1.ITvedant/2.ML projects/Silage_Sales_ML_Project/Pancham_Silage_Factory.csv"
INTO TABLE main_data
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES
(
Transaction_ID,
@Date,
Customer_Type,
Crop_Type,
Harvest_Season,
@Moisture,
@DM,
@Quantity,
@Price,
@TotalSales,
@Logistics,
@Profit,
Payment_Mode,
@CreditDays,
Bagging_Type
)

SET
-- ✅ Date conversion (change format if needed)
Date = STR_TO_DATE(NULLIF(@Date,''),'%d-%m-%Y'),

-- ✅ Remove % symbol + blanks
Moisture_Content_Percent =
    NULLIF(REPLACE(@Moisture,'%',''),''),

DM_Content_Percent =
    NULLIF(REPLACE(@DM,'%',''),''),

Profit_Margin_Percent =
    NULLIF(REPLACE(@Profit,'%',''),''),

-- ✅ Numbers (blank → NULL)
Quantity_MT = NULLIF(@Quantity,''),
Price_per_MT_INR = NULLIF(@Price,''),
Total_Sales_INR = NULLIF(@TotalSales,''),
Logistics_Cost_INR = NULLIF(@Logistics,''),

-- ✅ Integer safe conversion
Credit_Period_Days = NULLIF(@CreditDays,'');


    

SELECT COUNT(*) FROM main_data;
SELECT * FROM main_data LIMIT 5;