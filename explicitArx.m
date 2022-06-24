na = 3;
nb = ones(1,size(u,2));
nk = [0 0 0 0 0 0 0 0 0 0 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15];
mm = arx(z1,[na nb nk]);
figure
compare(z2,mm)