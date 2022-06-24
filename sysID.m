dataTable = readtable('data/mergedData.csv');
dataTable(:,[248:end]) = fillmissing(dataTable(:,[248:end]), 'linear');
dataTable(:,[46]) = fillmissing(dataTable(:,[46]), 'linear');
%dataTable(:,[49,51,53, 55]) = smoothdata(dataTable(:,[49,51,53,55]), 'movmean',20, 'omitnan');
%dataTable(:,[41,43,45,47]) = movmean(dataTable(:,[41,43,45,47]),30, 'omitnan');
data = [zeros(height(dataTable),1), table2array(dataTable(:,2:end))]; %compensate for timestamp label (+1 index)

timestamps =[1:height(dataTable)].';

xdirs = cos(pi*data(:,88:94)./180);
ydirs = sin(pi*data(:,88:94)./180);
salDiff = dataTable{:,333:2:343} - dataTable{:,265:2:275};

tideLevel = data(:,46);
currH = data(:,48:2:58);
currN = data(:,128:2:138);
currE = data(:,168:2:178);
currV = data(:,[208, 210, 218]);
salInside = data(:,[258,265,275]);
salOutside = data(:,[326, 336, 352]);
tempInside = data(:,[394]);

outputs = data(:,[38,40]);

data = [outputs, tideLevel, currH, currE, currN, currV, salInside, salOutside];
data(any(isnan(data), 2), :) = [];
u = data(:,[2]);%data(:,3:end);
y = data(:,[1]);
%y2 = data(:,[2]);
%u = u(:,fs);
z = iddata(y,u, 1, 'Name', 'DOTEMP');
z = detrend(z, 0);
%zval = iddata(y2,u, 1, 'Name', 'DOTEMP2');
z1 = z(1:2500);
z2 = z(2500:end);
%z1 = z([1,2,3,4,5]);
%z2 = z([6,7,8,9]);
% fs = [1 2 6 8 9 12 13 14 15 16 18 19 21 23 24 25 26 27 28];

%stackedplot(z2)
NN = struc(1:10, 1:10 , 1:15);
NN = [NN(:,1), repmat(NN(:,2),1,size(u,2)), repmat(NN(:,3),1,size(u,2))];
V = arxstruc(z1,z2, NN);
order = selstruc(V, 'aic');
m = arx(z1, order) % [3, 1, 2] > 50% [3, 1 ,9]
compare(z2,m)
figure
compare(z1,m)



%[3, 5, 19, 15, 5, 2, 2, 3, 2, 5, 1]