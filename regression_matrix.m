function x = regression_matrix(order,states)

x = zeros(0,states);
if states <=1  
for i=0:order
    if i<=order
        poly = [i];
        x =vertcat(x,poly); 
    end
end
else
    temp =regression_matrix(order,states-1);
    new_col =zeros(size(temp,1),1);
    x = horzcat(temp,new_col);
    for j =1:size(temp,1)
        if sum(temp(j,:)) < order
            value = order -sum(temp(j,:));
            for k= 1:value
            new_row = horzcat(temp(j,:),k);
            x = vertcat(x,new_row);
            end
        end
    end
end
end
            
    
    
  

        
    

    