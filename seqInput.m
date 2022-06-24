function fit = seqInput(X,Y)
    z = iddata(Y,X, 1, 'Name', 'DOTEMP');
    z = detrend(z, 0);
    z1 = z(1:2500);
    z2 = z(2500:end);
    
    NN = struc(1:10, 1:10 , 1:15);
    NN = [NN(:,1), repmat(NN(:,2),1,size(X,2)), repmat(NN(:,3),1,size(X,2))];
    V = arxstruc(z1,z2, NN);
    order = selstruc(V, 'aic');
    m = arx(z1, order);
    [~, fit, ic] = compare(z2,m);
    fit = -fit;
end
