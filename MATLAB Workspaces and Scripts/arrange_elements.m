function arr_elements = arrange_elements(geom_points, array_to_arrange, axis)

x_axis_length = length(unique(geom_points(:,1)));
y_axis_length = length(unique(geom_points(:,2)));
z_axis_length = length(unique(geom_points(:,3)));

j = 1;

if axis == 'x'
    arr_elements = zeros([y_axis_length * z_axis_length x_axis_length]);
    step = x_axis_length;
elseif axis == 'y'
    arr_elements = zeros([x_axis_length * z_axis_length y_axis_length]);
    step = y_axis_length;
else
    arr_elements = zeros([y_axis_length * x_axis_length z_axis_length]);
    step = z_axis_length;
end

for i = 1:step:length(array_to_arrange)
    arr_elements(j, :) = array_to_arrange(i:(i + step - 1));
    j = j + 1;
end
end