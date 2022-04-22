clc
clear
close all

x = audioread("dataAdaptiveFilter/talkingNoise.mp3");
[d, Fs] = audioread("dataAdaptiveFilter/bassLineTalkingNoise.mp3");
x = x(:,1);
d = d(:,1);
N = length(x);
time = (0:N-1)./Fs;
plot(time(100:200), x(100:200));
title("Talking noise signal: x[n]");
figure;
plot(time(100:200), d(100:200));
title("Bass Line Talking Noise Signal: d[n]");

% T = 3;
% f = 8;
% Fs = 8000;
% t = 0:1/Fs:T;
% N = length(t);
% 
% s = sin(2*pi*f*t);
% x = randn(1,N);
% h = [1 -0.8 0.6 -0.4 0.2];
% temp = conv(h, x);
% d = s + temp(1:N);
% 
% figure; plot(t,s); title("Source signal: s[n]");
% figure; plot(t, d); title("Received signal: d[n]");

K = 10;      % filter order
mu = 0.001;     % step size
f = zeros(1, K);
x_buffer = zeros(1, K);
e = zeros(1, N);
y = zeros(1, N);

for i=1:N
    x_buffer = [x(i) x_buffer(1:end-1)];
    y(i) = x_buffer*f';
    e(i) = d(i) - y(i);
    f = f + mu * conj(e(i)) * x_buffer;
end

figure;
plot(time, e);
title("Estimated Bass Line Signal");

figure;
% plot(time, s, 'linewidth', 2.0);
% hold on;
plot(time, e, '--');
legend("Source", "Estimated Source");

% figure; 
% plot(time, e-s);
% title("Estimation Error");
