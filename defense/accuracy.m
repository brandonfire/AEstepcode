M = csvread("attack.txt");
X = M(:,1);

Y1 = M(:,2);
Y2 = M(:,3);
Y3 = M(:,4);
Y4 = M(:,5);
%E = M(:,3);
plot(X,Y1);
hold on;
plot(X,Y2);
hold on;
plot(X,Y3);
hold on;
plot(X,Y4);
hold on;
plot(X,Y5);
%hold on;
%yyaxis left
%errorbar(X,Y,E);
%ecdf(X1,'Bounds','on');
%hold on;
%grid on;
%legend('Positioning Error CDF', 'Lower Confidence Bound', 'Upper Confidence Bound');
%hold off;

