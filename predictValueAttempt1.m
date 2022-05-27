data    = readmatrix('KOTAdataset.xlsx');
x       = fliplr(data(:,4:8)');
%x       = fliplr(data');
xtrain  = x(:,1:1500);
ytrain  = x(:,2:1501);

xtest   = x(:,1501:2075);
ytest   = x(:,1502:2076);

net     = feedforwardnet([]);
net     = configure(net,xtrain,ytrain);
net     = train(net,xtrain,ytrain);

    


u=net(xtest);

% plot(ytrain(1,:));hold on;
% plot(u(1,:),'r');
% legend('ytrain','xtrainNNoutput');
% title('1 hidden layer, 10 neuron (train set)');

plot(ytest(1,:));hold on;
plot(u(1,:),'r');

title('same nn 100 execution');
legend('ytest','xtestNNoutput');

