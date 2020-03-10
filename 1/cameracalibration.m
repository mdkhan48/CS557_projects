clear all
clc
format long
%image coordinate from 'getpoint'  %3d world coordinate from 'calibration-rig.jpg'
x1=842.1; y1=945.41; xw1=4; yw1=0; zw1=0;
x2=792.74; y2=745.74; xw2=6; yw2=0; zw2=6;
x3=571.23; y3=881.95; xw3=14; yw3=0; zw3=2;
%x4=846.739; y4=466.15; xw4=4; yw4=0; zw4=4; %near
%x5=743.039; y5=467.698; xw5=8; yw5=0; zw5=4; %near
x4=685.77; y4=457.86; xw4=10; yw4=0; zw4=14; %far
x5=571.23; y5=662.16; xw5=14; yw5=0; zw5=8; %far
x6=1078.9; y6=946.95; xw6=0; yw6=6; zw6=0;
x7=1030.9; y7=809.25; xw7=0; yw7=4; zw7=4;
x8=1286.4; y8=733.36; xw8=0; yw8=14; zw8=6;
%x9=1080.451; y9=469.246; xw9=0; yw9=6; zw9=4; %near
%x10=1182.603; y10=472.341; xw10=0; yw10=10; zw10=4; %near
x9=1225.9; y9=589.42; xw9=0; yw9=12; zw9=10; %far
x10=1337.4; y10=431.55; xw10=0; yw10=16; zw10=14; %far

%rearranging equations of slide 109
% -xi(a31*xwi+a32*ywi+a33*zwi)+(a11*xwi+a12*ywi+a13*zwi+a14)=xi*a34
% -yi(a31*xwi+a32*ywi+a33*zwi)+(a21*xwi+a22*ywi+a23*zwi+a14)=yi*a34
% setting a34=1
% recreate matrix using slide 111

A = [xw1 yw1 zw1 1 0 0 0 0 -x1*xw1 -x1*yw1 -x1*zw1;
     0 0 0 0 xw1 yw1 zw1 1 -y1*xw1 -y1*yw1 -y1*zw1;
     xw2 yw2 zw2 1 0 0 0 0 -x2*xw2 -x2*yw2 -x2*zw2;
     0 0 0 0 xw2 yw2 zw2 1 -y2*xw2 -y2*yw2 -y2*zw2;
     xw3 yw3 zw3 1 0 0 0 0 -x3*xw3 -x3*yw3 -x3*zw3;
     0 0 0 0 xw3 yw3 zw3 1 -y3*xw3 -y3*yw3 -y3*zw3;
     xw4 yw4 zw4 1 0 0 0 0 -x4*xw4 -x4*yw4 -x4*zw4;
     0 0 0 0 xw4 yw4 zw4 1 -y4*xw4 -y4*yw4 -y4*zw4;
     xw5 yw5 zw5 1 0 0 0 0 -x5*xw5 -x5*yw5 -x5*zw5;
     0 0 0 0 xw5 yw5 zw5 1 -y5*xw5 -y5*yw5 -y5*zw5;
     xw6 yw6 zw6 1 0 0 0 0 -x6*xw6 -x6*yw6 -x6*zw6;
     0 0 0 0 xw6 yw6 zw6 1 -y6*xw6 -y6*yw6 -y6*zw6;
     xw7 yw7 zw7 1 0 0 0 0 -x7*xw7 -x7*yw7 -x7*zw7;
     0 0 0 0 xw7 yw7 zw7 1 -y7*xw7 -y7*yw7 -y7*zw7;
     xw8 yw8 zw8 1 0 0 0 0 -x8*xw8 -x8*yw8 -x8*zw8;
     0 0 0 0 xw8 yw8 zw8 1 -y8*xw8 -y8*yw8 -y8*zw8;
     xw9 yw9 zw9 1 0 0 0 0 -x9*xw9 -x9*yw9 -x9*zw9;
     0 0 0 0 xw9 yw9 zw9 1 -y9*xw9 -y9*yw9 -y9*zw9;
     xw10 yw10 zw10 1 0 0 0 0 -x10*xw10 -x10*yw10 -x10*zw10;
     0 0 0 0 xw10 yw10 zw10 1 -y10*xw10 -y10*yw10 -y10*zw10];
 b = [x1;
      y1;
      x2;
      y2;
      x3;
      y3;
      x4;
      y4;
      x5;
      y5;
      x6;
      y6;
      x7;
      y7;
      x8;
      y8;
      x9;
      y9;
      x10;
      y10];
  %solving Ap=b
  p = A\b;
  
  %converting column vetor p to 3x4 projection matrix
  paddingvalue=1;
  projection_mat = vec2mat(p,4,paddingvalue)
  % First normalization slide 124
  %divide the pm with norm-transposed first three elements of row 3
  r3 = projection_mat(3,1:3)
  r3_T = transpose(r3);
  r3_T_norm = norm(r3_T);
  
  pm = projection_mat/r3_T_norm;
  
  r3new = pm(3,1:3);
  r3new*transpose(r3new);%check for orthonormality
  
  %dividing projection matring into B and b matrix, using slide 115
  B = [pm(1,1) pm(1,2) pm(1,3);
       pm(2,1) pm(2,2) pm(2,3);
       pm(3,1) pm(3,2) pm(3,3)];
  b = [pm(1,4);
       pm(2,4);
       pm(3,4)];
  % creating A=B*B^T
  A = B * transpose(B);
  %A(3,3)
  %normalizing A according to slide 116
  A_norm = A/A(3,3);

  
  %assume skew s = 0
  s = 0;
  %recovering intrinsic properties slide 119
  u0 = A_norm(1,3);
  v0 = A_norm(2,3);
  beta = sqrt(A_norm(2,2)-v0^2);
  alpha = sqrt(A_norm(1,1)-u0^2);
  
  %creating intrinsic matrix K
  fprintf('Intrinsic Matrix\n')
  K = [alpha s u0;
       0 beta v0;
       0 0 1]
   
   %Recovering extrinsic matrix from slide 121
   %rotation matrix
   fprintf('Rotation Matrix\n')
   R = K\B
   fprintf('check orthonormality\n R*R^T')
   R*transpose(R) %check orthonormality
   %translation matrix
   fprintf('Translation Matrix\n')
   t = K\b
   
   
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %          Error Calculation              %
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
   created_proj_mat = K*[R t] %this should be same as pm from line 68
    
   %Points near (0,0,0)
   %point 1                 %2D points from getpoint code(true values)
   xw1_c=4; yw1_c=0;zw1_c=0;     x1_c=843.6;y1_c=946.94;
   %point 2
   xw2_c=6; yw2_c=0;zw2_c=0;     x2_c=792.56;y2_c=946.9;
   %point3
   xw3_c=0; yw3_c=4;zw3_c=0;     x3_c=1032.5;y3_c=943.86;
   %point4
   xw4_c=0; yw4_c=4;zw4_c=2;     x4_c=1034;y4_c=878.8;
   
   %Points far from (0,0,0)
   %point5
   xw5_c=0; yw5_c=4;zw5_c=10;     x5_c=1027.8;y5_c=607.86;
   %point6
   xw6_c=10; yw6_c=0;zw6_c=10;     x6_c=687.3;y6_c=600.25;
   
   
   %calculating 2d point from proj_mat then compare with 2d points from
   %'getpoint'(true value) for error
   
   %point1
   point_1 = created_proj_mat*[xw1_c;
                               yw1_c;
                               zw1_c;
                               1];
   point1 = point_1/point_1(3,1);
   
   
   %point2
   point_2 = created_proj_mat*[xw2_c;
                               yw2_c;
                               zw2_c;
                               1];
   point2 = point_2/point_2(3,1);
   
   %point3
   point_3 = created_proj_mat*[xw3_c;
                               yw3_c;
                               zw3_c;
                               1];
   point3 = point_3/point_3(3,1);
   
   %check if original projection matrix works
   point_3_new = pm*[xw3_c;
                               yw3_c;
                               zw3_c;
                               1];
   point3new = point_3_new/point_3_new(3,1);
   
   %point4
   point_4 = created_proj_mat*[xw4_c;
                               yw4_c;
                               zw4_c;
                               1];
   point4 = point_4/point_4(3,1);
   
   %point5
   point_5 = created_proj_mat*[xw5_c;
                               yw5_c;
                               zw5_c;
                               1];
   point5 = point_5/point_5(3,1);
   %check point 5 original pm
   point_5_new = pm*[xw5_c;
                     yw5_c;
                     zw5_c;
                     1];
   point5new = point_5_new/point_5_new(3,1);
   
   %point6
   point_6 = created_proj_mat*[xw6_c;
                               yw6_c;
                               zw6_c;
                               1];
   point6 = point_6/point_6(3,1);
  %check if original projection matrix works
   point_6_new = pm*[xw6_c;
                     yw6_c;
                     zw6_c;
                     1];
   point6new = point_6_new/point_6_new(3,1);
   
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % point1 =        point2 =        point3                 point4

  %845.8517         795.1620        1.0e+03 *1.0338        1.0e+03 *1.0336
  %332.3617         330.5582        0.3341                  0.3271
   % 1.0000         1.0000          0.0010                  0.0010

   
%error in point 1 calculation
[point1_xerror,point1_yerror,norm1] = error(x1_c,point1(1,1),y1_c,point1(2,1));
X1 = sprintf('point 1 x-coordinate error %f%% and point 1 y-coordinate error %f%%. Normalized point 1 error %f%%',point1_xerror,point1_yerror,norm1);
disp(X1)

%error in point 2 calculation
[point2_xerror,point2_yerror,norm2] = error(x2_c,point2(1,1),y2_c,point2(2,1));
X2 = sprintf('point 2 x-coordinate error %f%% and point 2 y-coordinate error %f%%. Normalized point 2 error %f%%',point2_xerror,point2_yerror,norm2);
disp(X2)

%error in point 3 calculation
[point3_xerror,point3_yerror,norm3] = error(x3_c,point3(1,1),y3_c,point3(2,1));
X3 = sprintf('point 3 x-coordinate error %f%% and point 3 y-coordinate error %f%%. Normalized point 3 error %f%%',point3_xerror,point3_yerror,norm3);
disp(X3)
%error in point3new calculation
%[point3new_xerror,point3new_yerror,norm3new] = error(x3_c,point3new(1,1),y3_c,point3new(2,1));
%X3new = sprintf('point 3 new(org. pm) x-coordinate error %f%% and point 3 y-coordinate error %f%%. Normalized point 3 error %f%%',point3new_xerror,point3new_yerror,norm3new);
%disp(X3new)

%error in point 4 calculation
[point4_xerror,point4_yerror,norm4] = error(x4_c,point4(1,1),y4_c,point4(2,1));
X4 = sprintf('point 4 x-coordinate error %f%% and point 4 y-coordinate error %f%%. Normalized point 4 error %f%%',point4_xerror,point4_yerror,norm4);
disp(X4)

%error in point 5 calculation
[point5_xerror,point5_yerror,norm5] = error(x5_c,point5(1,1),y5_c,point5(2,1));
X5 = sprintf('point 5 x-coordinate error %f%% and point 5 y-coordinate error %f%%. Normalized point 5 error %f%%',point5_xerror,point5_yerror,norm5);
disp(X5)
%error in point5new calculation
%[point5new_xerror,point5new_yerror,norm5new] = error(x5_c,point5new(1,1),y5_c,point5new(2,1));
%X5new = sprintf('point 5 new(original pm) x-coordinate error %f%% and point 5 y-coordinate error %f%%. Normalized point 5 error %f%%',point5new_xerror,point5new_yerror,norm5new);
%disp(X5new)

%error in point 6 calculation
[point6_xerror,point6_yerror,norm6] = error(x6_c,point6(1,1),y6_c,point6(2,1));
X6 = sprintf('point 6 x-coordinate error %f%% and point 6 y-coordinate error %f%%. Normalized point 6 error %f%%',point6_xerror,point6_yerror,norm6);
disp(X6)
%error in point6new calculation
%[point6new_xerror,point6new_yerror,norm6new] = error(x6_c,point6new(1,1),y6_c,point6new(2,1));
%X6new = sprintf('point 6 new(original pm) x-coordinate error %f%% and point 6 y-coordinate error %f%%. Normalized point 6 error %f%%',point6new_xerror,point6new_yerror,norm6new);
%disp(X6new)
error_norm = [norm1 norm2 norm3 norm4 norm5 norm6];
summary(error_norm)

function [Xerror,Yerror,err_norm] = error(x_true,x_measured,y_true,y_measured)
Xerror = 100*abs((x_true-x_measured)/x_true);
Yerror = 100*abs((y_true-y_measured)/y_true);
err_norm = sqrt(Xerror^2 + Yerror^2);
end

function [maxout, minout, meanout, medianout, stdout] = summary(error_norm)

maxout    = max(error_norm);
minout    = min(error_norm);
meanout   = mean(error_norm);
medianout = median(error_norm);
stdout    = std(error_norm);
fprintf ('Error : Maximum %f%%, Minimum %f%%, Mean %f, Median %f, Std Dev %f',maxout,minout,meanout,medianout,stdout)
end

