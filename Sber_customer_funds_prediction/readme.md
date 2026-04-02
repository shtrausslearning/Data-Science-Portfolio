## **1. Project Background**


Our department is engaged in liquidity risk management, namely, ensuring the necessary level of funds to achieve Sberbank's strategic business development goals.
As an intern in our department, your task today will be to learn how to predict the volume of stable client funds without maturities, in this particular case, these are client settlement accounts.

Why is it important? Nominally, all funds on current accounts can be withdrawn from the Bank at any time, and in anticipation of this, the Bank cannot use them in the long / medium term (for example, for issuing loans). It turns out that in such a situation the Bank does not earn anything, but pays customers interest on the funds in their accounts, although not high, but on the scale of the Bank's business, these losses can be significant.

But in reality, customer behavior is different, it depends on many factors (behavioral, macroeconomic, competitors' actions, etc.). Clients do not immediately withdraw all their funds from settlement accounts, but keep them there for some time, therefore, in total for all clients, their settlement accounts always have some amount of funds, which, although it changes over time, can be regarded by the Bank as stable and used for issuing loans (and the Bank makes money on this).

The ability to accurately predict the volume and dynamics of a stable balance of funds on current accounts allows the Bank to earn on lending, but at the same time keep the risk that customers can demand these funds back at any time – this is called "liquidity risk management". To do this, an ML model is built to predict a stable balance of funds on customer settlement accounts, associated with models for forecasting markets, macroeconomics, and customer behavior.

The scale of the Bank's business is amazing: for key banking products, the share reaches 30-50%, which means tens of millions of customers, trillions of rubles in volume. Increasing the forecast accuracy of just one such model, for example, by 5% or in terms of money by 50 billion rubles, will allow the Bank to earn an additional 1 billion rubles. per year (assuming 5% margin of the banking business).
