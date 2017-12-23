% DIGITAL IMAGE PROCESSING
% Time behaviour of convolution by usage of spatial and frequency domain

clc;
clear all;
close all;
% Loading  Frequency 
con_fre_256 = load('convolutionFrequencyDomain_256.txt');
con_fre_384 = load('convolutionFrequencyDomain_384.txt'); 
con_fre_512 = load('convolutionFrequencyDomain_512.txt');
con_fre_768 = load('convolutionFrequencyDomain_768.txt');
con_fre_1024 = load('convolutionFrequencyDomain_1024.txt');
% Merging to one matrix
con_fre_all = [con_fre_256 con_fre_384 con_fre_512 con_fre_768 con_fre_1024];
% Loading Spatial
con_spa_256 = load('convolutionSpatialDomain_256.txt');
con_spa_384 = load('convolutionSpatialDomain_384.txt');
con_spa_512 = load('convolutionSpatialDomain_512.txt');
con_spa_768 = load('convolutionSpatialDomain_768.txt');
con_spa_1024 = load('convolutionSpatialDomain_1024.txt');
% Merging to one matrix
con_spa_all = [con_spa_256 con_spa_384 con_spa_512 con_spa_768 con_spa_1024];
% Kernel Matrix of Different Size
kernel_size = load('kernel_size.txt');
% Loading Resolution Matrix
resolution = load('resolution.txt');
%Surface plot of Frequency Domain
figure;
grid
surf(kernel_size,resolution', con_fre_all')
colorbar
caxis([0,0.5]);
xlabel('Kernel Size','FontSize',15)
ylabel('Resolution','FontSize',15)
zlabel('Frequency Domain Time','FontSize',15)
title('Frequency Domain Plot','FontSize',15)

%Surface plot of Spatial Domain
figure;
grid
surf(kernel_size,resolution',con_spa_all');
colorbar
caxis([0,200]);
xlabel('Kernel Size','FontSize',15)
ylabel('Resolution','FontSize',15)
zlabel('Spatial Domain Time','FontSize',15)
title('Spatial Domain Plot','FontSize',15)
