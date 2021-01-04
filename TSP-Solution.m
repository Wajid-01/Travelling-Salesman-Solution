clc; clear all; close all;

%% [Step 0] -  Choose task 1 or 3 (for task 2, variables have been changed and plotted in the report):
workshop_task = 1;
%workshop_task = 3;

switch workshop_task
    case 1
%% [Step 1] -  Create Randomly Located Cities within the range: 0 - 10:

no_city = input('Please Choose Number of Cities the Salesman Travels to: '); %Define number of cities you want the Kohonen algorithm to solve for
x_city = 0 + (10)*rand(1,no_city); y_city = 0 + (10)*rand(1,no_city);

%% [Step 2] - Randomise Weights for 1-D Solution:

x_weights = 0; y_weights = 0;

%% [Step 3] - Define and Pre-Set Parameter (& other) Values for the Kohonen Network:
learning_rate = 0.01; n_start = 30; % n_start = starting neighbour radius
Network_memory = 0; Network_generation = 0;
Network_iterations = [0 500];
iterations = Network_iterations(1); %number of initial iterations or starting time = 0;
max_iterations = Network_iterations(2);%maximum number of iterations (after 500 network makes extremely little (or no) progress - hence 500 
a_1 = 1; a_3 = 3;

%% [Step 4] - Run simulation to solve TSP:

while iterations < max_iterations
    
   n_value = []; iterations = iterations+1;
   
    for index_DD = a_1 :length(Network_memory) %wipe network memory
       Network_memory(index_DD) = 0;
    end
    
    n_start(iterations+1)=(1-learning_rate)*n_start(iterations); 
    
    for t_index = a_1:no_city;
    duplicate_node = false;
    
    %Section 1: search for the closest node to the city @ iteration n (or time t) - Euclidian Distance: 
    Euclidian_distance = (x_city(t_index)-x_weights).^2+(y_city(t_index)-y_weights).^2; 
            x_minimum = 1; minimum_distance = Euclidian_distance(x_minimum);    
            for n_j1 = a_1:size(x_weights,2)
                    if Euclidian_distance(n_j1) < minimum_distance
                        minimum_distance = Euclidian_distance(n_j1);
                        x_minimum = n_j1;
                    end
            end
            n_j1star = x_minimum;
    
         %Section 2: Create and move smallest node to closest city
               if Network_memory(n_j1star) >= 1  % if the network searches for the same (identical) node, then duplicate the node
                  duplicate_node = true;
                  
                  x_weights = [x_weights(1:n_j1star) x_weights(n_j1star) x_weights(n_j1star+1:end)];
                  y_weights = [y_weights(1:n_j1star) y_weights(n_j1star) y_weights(n_j1star+1:end)];
                  
                  Euclidian_distance = [Euclidian_distance(1:n_j1star) Euclidian_distance(n_j1star) Euclidian_distance(n_j1star+1:end)];
                  
                  Network_memory = [Network_memory(1:n_j1star) 0 Network_memory(n_j1star+1:end)];
                  Network_generation = [Network_generation(1:n_j1star) 0 Network_generation(n_j1star+1:end)];
                  
                  n_j1star = n_j1star+1; %re-produce the highlighted node (duplicate)
               end
                      
            %update the duplicated node
            for gen_idx = a_1 : size(x_weights,2)  
                n_value(gen_idx) = min (mod((gen_idx-n_j1star),size(x_weights,2)),mod((n_j1star-gen_idx),size(x_weights,2)));
            end
            f_value = (1/sqrt(2))*exp(-(n_value.^2)/n_start(iterations)^2);
            
            % non-inhibited node is updated in this context solely (e.g. n_value~=0)
            for gen2_index = a_1 : size(x_weights,2) 
                if duplicate_node == true
                    if gen2_index ~= n_j1star || n_j1star-1
                        x_weights(gen2_index) = x_weights(gen2_index)+f_value(gen2_index)*(x_city(t_index)- x_weights(gen2_index));
                        y_weights(gen2_index) = y_weights(gen2_index)+f_value(gen2_index)*(y_city(t_index)- y_weights(gen2_index)); 
                    end
                else
                    x_weights(gen2_index) = x_weights(gen2_index)+f_value(gen2_index)*(x_city(t_index)- x_weights(gen2_index));
                    y_weights(gen2_index) = y_weights(gen2_index)+f_value(gen2_index)*(y_city(t_index)- y_weights(gen2_index)); 
                end
            end

            Network_memory(n_j1star) =  Network_memory(n_j1star) + 1;
            
    end
    
    net_ones = ones(size(Network_generation));
    Network_generation= Network_generation + net_ones;
    Network_generation(Network_memory > 0) = 0;
    x_weights(Network_generation > a_3) = [];
    y_weights(Network_generation > a_3) = [];
    Euclidian_distance(Network_generation > a_3) = []; 
    Network_memory(Network_generation > a_3) = []; 
    Network_generation(Network_generation > a_3) = [];
  
  
  
%% Step [5] - Visualise How the Kohonen Network Solves the TSP (Plot Results):

plot([x_weights x_weights],[y_weights y_weights],'k','linewidth',2.5);
hold on; 
plot(x_weights,y_weights,'bo','MarkerSize',5);
plot(x_city,y_city,'r.','MarkerSize',8); 
hold off
title(['Kohonen TSP Solver (1-D Line): ' 'No. of Iterations=' num2str(iterations)]);
xlabel('X Coordinate (City)'); ylabel('Y Coordinate (City)');
drawnow % NOTE: comment this line, if you want to skip simulation and plot final graph (saves time!)
end

legend('Salesman Path', 'Updating Neuron(s) Weights', 'Independant City');

    case 3      
%% [Step 1] -  Create Randomly Located Cities within the range: 0 - 10

no_city = input('Please Choose Number of Cities the Salesman Travels to: '); %Define number of cities you want the Kohonen algorithm to solve for
x_city = 0 + (10)*rand(1,no_city); y_city = 0 + (10)*rand(1,no_city);

%% [Step 2] - Randomise Weights for 2-D/3-D Solution:

x_weights = randn(10,10); y_weights = randn(10,10); %contrary to the 1-D network, weights are multidimensional (10 x 10 matrix of coordinates)

%% [Step 3] - Define and Pre-Set Parameter (& other) Values for the 2-D Kohonen Network:

scan_neighbor = 4; %input number of neighbours the network accounts for (this network is restricted to 4 as requested by workshop guidelines)
gain = 1; grid_region = 5; %network gain and grid region the kohonen algorithm accounts for
nj1_index = 10; nj2_index = 10;
iterations=0;

%% [Step 4] - Run simulation to solve TSP:

while iterations < no_city
    
    %Section 1: find distance and update weights:
    iterations=iterations+1;
    n_value = gain*(1-iterations/no_city); %update thenetwork gain as iterations pass (relative to iterations(time))
    D_value = round(grid_region*(1-iterations/no_city));  %update the values of the neighbour nodes (up,down,left & right) relative to iterations (time)
    Euclidian_distance = (x_city(iterations)-x_weights).^2+(y_city(iterations)-y_weights).^2;
        
        if scan_neighbor ~= 4          
            
            sound_amp = 10; fs = 15000; timing = 0.08;
            frequency = 50; values = 0:1/fs:timing;
            error_sound = sound_amp*sin(5*pi* values*frequency);
            sound(error_sound);
            disp('Error, Please Reset Neighbour Value to 4!');           
            break
            
        else    
            x_minimum = 1; y_minimum = 1; minimum_distance = Euclidian_distance(x_minimum,y_minimum);  
            
            for n_j1 = 1:nj1_index;
                for n_j2 = 1:nj2_index;
                    if Euclidian_distance(n_j1,n_j2) < minimum_distance
                        minimum_distance = Euclidian_distance(n_j1,n_j2); x_minimum = n_j1; y_minimum = n_j2;
                    end
                end
            end
            n_j1star= x_minimum; n_j2star= y_minimum;
            
            % Section 2: Apply weight updates to winning neuron in kohonen network (appropriate x/y coordinates for x/y weights       
            x_weights(n_j1star,n_j2star)=x_weights(n_j1star,n_j2star)+n_value*(x_city(iterations)- x_weights(n_j1star,n_j2star)); %update for x coordinates of winning neuron
            y_weights(n_j1star,n_j2star)=y_weights(n_j1star,n_j2star)+n_value*(y_city(iterations)- y_weights(n_j1star,n_j2star)); %update for y coordinates of winning neuron
        end
        
        %Section 3: update weights for the neighbour neurons (up, left, right, down):                
            for index_DD=1:1:D_value
                n_jj1=n_j1star-index_DD; n_jj2=n_j2star;
                if (n_jj1>=1)
                    x_weights(n_jj1,n_jj2)=x_weights(n_jj1,n_jj2)+n_value*(x_city(iterations)-x_weights(n_jj1,n_jj2));
                    y_weights(n_jj1,n_jj2)=y_weights(n_jj1,n_jj2)+n_value*(y_city(iterations)-y_weights(n_jj1,n_jj2));
                end
                
                n_jj1=n_j1star+index_DD; n_jj2=n_j2star;
                if (n_jj1<=10)
                    x_weights(n_jj1,n_jj2)=x_weights(n_jj1,n_jj2)+n_value*(x_city(iterations)-x_weights(n_jj1,n_jj2));
                    y_weights(n_jj1,n_jj2)=y_weights(n_jj1,n_jj2)+n_value*(y_city(iterations)-y_weights(n_jj1,n_jj2));
                end
                n_jj1=n_j1star; n_jj2=n_j2star-index_DD;
                if (n_jj2>=1)
                    x_weights(n_jj1,n_jj2)=x_weights(n_jj1,n_jj2)+n_value*(x_city(iterations)-x_weights(n_jj1,n_jj2));
                    y_weights(n_jj1,n_jj2)=y_weights(n_jj1,n_jj2)+n_value*(y_city(iterations)-y_weights(n_jj1,n_jj2));
                end
                
                n_jj1=n_j1star; n_jj2=n_j2star+index_DD;
                if (n_jj2<=10)
                    x_weights(n_jj1,n_jj2)=x_weights(n_jj1,n_jj2)+n_value*(x_city(iterations)-x_weights(n_jj1,n_jj2));
                    y_weights(n_jj1,n_jj2)=y_weights(n_jj1,n_jj2)+n_value*(y_city(iterations)-y_weights(n_jj1,n_jj2));
                end
            end             
        end
  
 %% [Step 5] - Visualise the 2-D Kohonen TSP Solver (Plot Results):
 
    if iterations == no_city       
                
                plot(x_city(1:iterations),y_city(1:iterations),'.b')
                hold on
                plot(x_weights,y_weights,'or')
                plot(x_weights,y_weights,'k','linewidth',2)
                plot(x_weights',y_weights','k','linewidth',2)
                hold off
                title(['Kohonen TSP Solver (2-D Grid):' 'No. of Iterations=' num2str(iterations)]); 
                xlabel('X Coordinate (City)'); ylabel('Y Coordinate (City)');               
    end
end