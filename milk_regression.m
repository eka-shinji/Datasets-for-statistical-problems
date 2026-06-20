% milk_regression.m
% Analysis of milk production and milk protein production
clear
clc
close all
%% Part (a): Import the data and make a scatterplot
% Import data from GitHub
url = "https://raw.githubusercontent.com/eka-shinji/" + ...
"Datasets-for-statistical-problems/main/milk.csv";
milkTable = readtable(url);

% Define x and y
x = milkTable.x; % milk production in kg/day
y = milkTable.y; % milk protein production in kg/day
% Also store the data as a matrix called milk
milk = [x y];
% Scatterplot
figure
scatter(x,y,'filled')
xlabel('Milk production, x (kg/day)')
ylabel('Milk protein production, y (kg/day)')
title('Protein production versus milk production')
grid on
%% Part (b): Fit the simple linear regression model
% Fit linear regression model
mdl = fitlm(x,y);
% Display model summary
disp(mdl)
% Extract fitted coefficients
b0 = mdl.Coefficients.Estimate(1);
b1 = mdl.Coefficients.Estimate(2);
fprintf('Fitted regression line:\n')
fprintf('yhat = %.4f + %.5f x\n', b0, b1)
%% Add least squares regression line to scatterplot
hold on
%plot(x, fitted(mdl), 'LineWidth', 2)
plot(x, mdl.Fitted, 'LineWidth', 2)
legend('Observed data','Least squares regression line', ...
'Location','northwest')
hold off
%% Estimated standard deviation of error term
sigmahat = mdl.RMSE;
fprintf('Estimated sigma = %.4f\n', sigmahat)
%% 95 percent confidence interval for beta_1
CI_beta = coefCI(mdl,0.05);
CI_beta1 = CI_beta(2,:);
fprintf('95%% confidence interval for beta_1:\n')
fprintf('(%.5f, %.5f)\n', CI_beta1(1), CI_beta1(2))
%% Test significance of predictor X
p_beta1 = mdl.Coefficients.pValue(2);
t_beta1 = mdl.Coefficients.tStat(2);
fprintf('Test statistic for beta_1 = %.4f\n', t_beta1)
fprintf('p-value for beta_1 = %.6f\n', p_beta1)
if p_beta1 < 0.05
fprintf('Reject H0: X is a significant predictor of Y.\n')
else
fprintf('Do not reject H0: X is not a significant predictor of Y.\n')
end
%% 99 percent confidence interval for mean response at x0 = 30
x0 = 30;
[ypred_mean, yci_mean] = predict(mdl, x0, 'Alpha', 0.01);
fprintf('Predicted mean protein production at x = 30:\n')
fprintf('yhat = %.4f\n', ypred_mean)
fprintf('99%% confidence interval for mean response at x = 30:\n')
fprintf('(%.4f, %.4f)\n', yci_mean(1), yci_mean(2))
%% 99 percent prediction interval for a single cow at x0 = 30
[ypred_obs, ypi_obs] = predict(mdl, x0, ...
'Alpha', 0.01, ...
'Prediction', 'observation');
fprintf('99%% prediction interval for a single cow at x = 30:\n')
fprintf('(%.4f, %.4f)\n', ypi_obs(1), ypi_obs(2))
%% Diagnostic plots
% Residuals versus fitted values
figure
plotResiduals(mdl,'fitted')
xlabel('Fitted values')
ylabel('Residuals')
title('Residuals versus fitted values')
grid on
% Normal quantile plot of residuals
figure
plotResiduals(mdl,'probability')
title('Normal probability plot of residuals')
grid on