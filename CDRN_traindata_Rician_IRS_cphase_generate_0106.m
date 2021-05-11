clc;
clear;

M = 8;              %number of antennas at BS 
N = 32;             %number of elements at IRS
C = N+1;            %number of reflection patterns

j = sqrt(-1);
datalength_train = 60000;

Np = 20;                         %number of parfors
Nd = datalength_train / Np;      %number of examples per parfor

SNRdB = 5 ;
varNoise = 1; 

Nt = N+1;           %number of columns of H = [d B]

pl_f = 0.18^-2 / 31.6;
pl_G = 1/31.6;
pl_d = 1/33;

K_dB = 10 ;
K = 10^(K_dB/10) ;

Wn = exp ( j* 2*pi / C ) ; 
temp = 0:C-1;
s = [];

for tt = 0 : Nt - 1

    P_row = Wn.^(  tt * temp );

    s = [s; P_row];

end

parfor ii = 1 : Np
    
   tic

   data_SNR_train = [];
   label_SNR_train = [];
   
   count = 0;
    
while count < datalength_train / Np 

        count = count + 1 ;
                
        f = sqrt(pl_f) * 1/sqrt (2)* ( normrnd( 0,1,N,1 ) + j*normrnd( 0,1,N,1 ) );  % channel user to IRS
                
        G = sqrt(pl_G) * ( sqrt(K/(K+1)) + sqrt(1/(K+1)) * 1/sqrt (2)* ( normrnd( 0,1,M,N ) + j*normrnd( 0,1,M,N ) ) ) ;  %IRS to BS
                
        B = G * diag ( f );  %cascaded channel user to IRS to BS
        
        d = sqrt(pl_d) * 1/sqrt (2) * ( normrnd( 0,1,M,1 ) + j*normrnd( 0,1,M,1 ) );  % channel user to BS
        
        H =  [d B];  %total channel  
             
        Power = 10^( SNRdB / 10 ) * varNoise;  %transmission power
        
        S = sqrt( Power ) * s ;
        
        W =  sqrt( varNoise ) * 1/sqrt(2) * ( normrnd( 0, 1, M, C ) + j*normrnd( 0, 1, M, C ) ); 
        
        X = H*S + W ;   %received signal at BS 
        
        %LS Detection
        
        w_ls = S' / ( S * S' );
                       
        h_est_ls = X * w_ls;
        
        data_SNR_train( count, :, :, 1 ) = real ( h_est_ls ) ;
        data_SNR_train( count, :, :, 2 ) = imag ( h_est_ls ) ;
        label_SNR_train( count, :, :, 1 ) = real ( H ) ;  
        label_SNR_train( count, :, :, 2 ) = imag ( H ) ; 
                       
end

file_px = 'x_train_Rician_CSCG_';  
file_py = 'y_train_Rician_CSCG_';


file_x_train = strcat( file_px, num2str(ii),'_M',num2str(M), '_N',num2str(N),'_', num2str(SNRdB),'dB.mat' );
file_h_train = strcat( file_py, num2str(ii),'_M',num2str(M), '_N',num2str(N),'_', num2str(SNRdB),'dB.mat' );

m1=matfile( file_x_train,'writable',true );
m1.x_subtrain =data_SNR_train;

m2=matfile( file_h_train,'writable',true );
m2.y_subtrain = label_SNR_train;

toc
	
end

x_train = [];
y_train = [];
file_px = 'x_train_Rician_CSCG_';  
file_py = 'y_train_Rician_CSCG_';
file_px_temp = 'x_train_Rician_CSCG_K'; 
file_py_temp = 'y_train_Rician_CSCG_K';
file_px_res = strcat( file_px_temp, num2str(K_dB), 'dB_' );
file_py_res = strcat( file_py_temp, num2str(K_dB), 'dB_' );



for tt = 1 : Np
        
    temp1 = 1 + ( tt-1 ) * Nd;
    temp2 = Nd + ( tt-1 ) * Nd;
    
    temp = temp1 : temp2;
    
    file_x_train = strcat( file_px, num2str(tt), '_M',num2str(M), '_N',num2str(N),'_', num2str(SNRdB),'dB.mat');  
    load( file_x_train );
    x_train( temp,:,:,: ) = x_subtrain;    
    
    file_y_train = strcat( file_py, num2str(tt), '_M',num2str(M), '_N',num2str(N),'_', num2str(SNRdB),'dB.mat');  
    load( file_y_train );
    y_train( temp,:,:,: ) = y_subtrain;         

    delete( file_x_train );
    delete( file_y_train );
    
end


file_x_train_res = strcat( file_px_res, num2str( datalength_train ), '_M',num2str(M), '_N',num2str(N),'_', num2str(SNRdB),'dB.mat' );   
save(file_x_train_res,'x_train');

file_y_train_res = strcat( file_py_res, num2str( datalength_train ), '_M',num2str(M), '_N',num2str(N),'_', num2str(SNRdB),'dB.mat' );   
save(file_y_train_res,'y_train');

    
disp('traindata generated!')