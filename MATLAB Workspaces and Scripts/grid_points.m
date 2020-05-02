% tic
% j = 1;
% geometry_points = zeros([(length(Axis2) - 1)*(length(Axis1) - 1)*(length(Axis0) - 1) 3]);
% 
% for z = 1:(length(Axis2) - 1)
%     for y = 1:(length(Axis1) - 1)
%         for x = 1:(length(Axis0) - 1)
%             geometry_points(j, :) = [(Axis0(x) + Axis0(x + 1))/2.0 (Axis1(y) + Axis1(y + 1))/2.0 (Axis2(z) + Axis2(z + 1))/2.0];
%             j = j + 1;
%         end
%    end
% end
% toc

tic
x_avg = 0.5*(Axis0(1:(length(Axis0) - 1)) + Axis0(2:end));
y_avg = 0.5*(Axis1(1:(length(Axis1) - 1)) + Axis1(2:end));
z_avg = 0.5*(Axis2(1:(length(Axis2) - 1)) + Axis2(2:end));

[X, Y, Z] = meshgrid(x_avg, y_avg, z_avg);
X = permute(X, [2 1 3]);
Y = permute(Y, [2 1 3]);
geom_points = [X(:) Y(:) Z(:)];
toc

clear x y z j;
%clear X Y Z x_avg y_avg z_avg;