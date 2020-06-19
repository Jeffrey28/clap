%% created by Zhong 2019/5/6: draw hist of visited states

clear all
close all
clc

% self.state = r_ego, Carla_state.ego_speed,\
%              Carla_state.front_vehicle_inside_distance,\
%              Carla_state.front_vehicle_inside_direction,\
%              Carla_state.front_vehicle_inside_speed,\
%              Carla_state.front_vehicle_outside_distance,\
%              Carla_state.front_vehicle_outside_direction,\
%              Carla_state.front_vehicle_outside_speed,\
%              Carla_state.behind_vehicle_inside_distance,\
%              Carla_state.behind_vehicle_inside_speed,\
%              Carla_state.behind_vehicle_outside_distance,\
%              Carla_state.behind_vehicle_outside_speed

load visited_state.txt
visited_state = visited_state;

r_ego = visited_state(:,2);
ego_speed = visited_state(:,3)*3.6;

d = sum(ego_speed/3.6*0.2)
% front_vehicle_inside_distance = visited_state(:,3);
% front_vehicle_inside_direction = visited_state(:,4);
% front_vehicle_inside_speed = visited_state(:,5);
% front_vehicle_outside_distance = visited_state(:,6);
% front_vehicle_outside_direction = visited_state(:,7);
% front_vehicle_outside_speed = visited_state(:,8);
% behind_vehicle_inside_distance = visited_state(:,9);
% behind_vehicle_inside_speed = visited_state(:,10);
% behind_vehicle_outside_distance = visited_state(:,11);
% behind_vehicle_outside_speed = visited_state(:,12);

plot(r_ego,ego_speed,'.')

figure
X = [r_ego,ego_speed];
X = sortrows(X,2);
X = X(2458:209474,:);

hist3(X,'Ctrs',{-0.5:0.01:1.5 0:0.1:30})
xlabel('r ego(m)')
ylabel('ego speed(km/h)')
zlabel('visited times')
% surfHandle = get(gca, 'child');
% set(surfHandle,'FaceColor','interp', 'CdataMode', 'auto');
% colorbar
% view(2)

%%
% min_componets = 2;
% GMMmodel = fitgmdist(X,2);
% min_AIC = GMMmodel.AIC;

% for i = 3:30
%     GMMmodel = fitgmdist(X,i);
%     i
%     if GMMmodel.AIC < min_AIC 
%         min_componets = i;
%         min_AIC = GMMmodel.AIC;
%     end 
% end
%%
GMMmodel = fitgmdist(X,40, 'Regularize',1e-5);
% 'CovType','diagonal','SharedCov',true
sigma = GMMmodel.Sigma;
mu = GMMmodel.mu;
%%
X1 = X(1,:);
Y = pdf(GMMmodel,[X1(1),X1(2)]);
[X1,Y1] = meshgrid(-0.5:0.05:1.5,0:0.1:30);
Z = X1;
C = X1;
for i=1:size(X1,1)
    for j=1:size(X1,2)
       Z(i,j) = pdf(GMMmodel,[X1(i,j),Y1(i,j)]);
       C(i,j) = Z(i,j)*100;
    end 
end
figure
surf(X1,Y1,Z)
xlabel('ego_lane')
ylabel('ego speed(km/h)')
zlabel('probability density')
% colorbar

%%
% figure
% [X,Y] = meshgrid(1:0.5:10,1:20);
% Z = sin(X) + cos(Y);
% C = X.*Y;
% surf(X,Y,Z,C)
% colorbar

%%

