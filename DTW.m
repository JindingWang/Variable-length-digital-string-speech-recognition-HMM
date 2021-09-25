function min_cost = DTW(template, data)
template_row = size(template,1);
data_row = size(data,1);
current = ones(1,template_row)*Inf;
next = zeros(1,template_row);
current(1) = sqrt(sum((template(1,:)-data(1,:)).^2));
for i = 1:data_row-1
    next(1) = current(1) + sqrt(sum((template(1,:)-data(i+1,:)).^2));
    next(2) = min(current(1), current(2))+sqrt(sum((template(2,:)-data(i+1,:)).^2));
    node_score = sqrt(sum((template(3:end,:)-repmat(data(i+1,:),template_row-2,1)).^2,2));
    next(3:template_row) = min([current(3:template_row); current(2:template_row-1); current(1:template_row-2)])+node_score';
    current = next;
end
min_cost = current(template_row);