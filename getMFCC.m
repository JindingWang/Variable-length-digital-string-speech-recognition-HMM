function mfcc = getMFCC(wavName)
[speech, Fs] = audioread(wavName); % read a speech
speech = filter([1,-0.9375], 1, speech);  % preemphasized
frame = enframe(speech, 256, 80); % 帧长256，帧移80
Dimension = 13;
bank = melbankm(24, 256, Fs, 0, 0.4 ,'t'); % Mel滤波器的阶数为26，fft变换的长度为256，采样频率为16kHz
bank = full(bank);
bank = bank / max(bank(:)); % 归一化mel滤波器组系数

% DCT系数
dctcoef = zeros(15,24);
m = 0:23;
for n = 1:15
    dctcoef(n,:) = cos((m+0.5)*n*pi/24);
end
w=1+7.5*sin(pi*[1:15]./15);%归一化倒谱提升窗口
w=w/max(w);%预加重滤波器

% 计算每帧的MFCC参数
framelength = size(frame,1);
mfc0 = zeros(framelength,Dimension);
for i = 1:framelength
    y = frame(i,:);
    s = y' .* hamming(256);
    t = abs(fft(s)).^2;
    c1 = dctcoef * log(bank*t(1:129));
    c2=c1.*w';
    mfc0(i,:) = c2(2:14);
end
% CMVN
% mean = sum(mfc0,1)/framelength;
% Xshift = mfc0 - repmat(mean, framelength, 1);
% variance = sqrt(sum(Xshift.^2,1)/framelength);
% mfc0 = Xshift./repmat(variance, framelength, 1);
%求MFCC的一阶差分
mfc1=zeros(size(mfc0));
for i=3:size(mfc0,1)-2
    mfc1(i,:)=-2*mfc0(i-2,:)-mfc0(i-1,:)+mfc0(i+1,:)+2*mfc0(i+2,:);
end
mfc1=mfc1/3;
%求MFCC的一阶差分
mfc2=zeros(size(mfc1));
for i=3:size(mfc1,1)-2
    mfc2(i,:)=-2*mfc1(i-2,:)-mfc1(i-1,:)+mfc1(i+1,:)+2*mfc1(i+2,:);
end
mfc2=mfc2/3;
%合并MFCC各阶参数
mfcc = [mfc0, mfc1, mfc2];
mfcc = mfcc(3:size(mfcc,1)-2,:);
[a,~] = find(isnan(mfcc)==1);
mfcc(a,:) = [];
end