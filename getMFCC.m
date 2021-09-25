function mfcc = getMFCC(wavName)
[speech, Fs] = audioread(wavName); % read a speech
speech = filter([1,-0.9375], 1, speech);  % preemphasized
frame = enframe(speech, 256, 80); % ֡��256��֡��80
Dimension = 13;
bank = melbankm(24, 256, Fs, 0, 0.4 ,'t'); % Mel�˲����Ľ���Ϊ26��fft�任�ĳ���Ϊ256������Ƶ��Ϊ16kHz
bank = full(bank);
bank = bank / max(bank(:)); % ��һ��mel�˲�����ϵ��

% DCTϵ��
dctcoef = zeros(15,24);
m = 0:23;
for n = 1:15
    dctcoef(n,:) = cos((m+0.5)*n*pi/24);
end
w=1+7.5*sin(pi*[1:15]./15);%��һ��������������
w=w/max(w);%Ԥ�����˲���

% ����ÿ֡��MFCC����
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
%��MFCC��һ�ײ��
mfc1=zeros(size(mfc0));
for i=3:size(mfc0,1)-2
    mfc1(i,:)=-2*mfc0(i-2,:)-mfc0(i-1,:)+mfc0(i+1,:)+2*mfc0(i+2,:);
end
mfc1=mfc1/3;
%��MFCC��һ�ײ��
mfc2=zeros(size(mfc1));
for i=3:size(mfc1,1)-2
    mfc2(i,:)=-2*mfc1(i-2,:)-mfc1(i-1,:)+mfc1(i+1,:)+2*mfc1(i+2,:);
end
mfc2=mfc2/3;
%�ϲ�MFCC���ײ���
mfcc = [mfc0, mfc1, mfc2];
mfcc = mfcc(3:size(mfcc,1)-2,:);
[a,~] = find(isnan(mfcc)==1);
mfcc(a,:) = [];
end