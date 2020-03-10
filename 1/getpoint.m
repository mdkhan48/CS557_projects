calling=imread('calibration-rig.jpg');
%calling=imread('closeup.jpg');
[rows columns depth]= size(calling)
f1=figure;
imshow(calling);
[x,y] = getpts(f1);
fid = fopen('img_pts.txt','w');
%pts=[x rows-y]';
%without row normalize
pts=[x y]';
fprintf(fid,'%f %f\s',pts);