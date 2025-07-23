Allostery in Proteins: Quantifying Allosteric
Communication in Ubiquitin
clear all
close all
atomno=404;
xyz= readdcd('200ns_ca.dcd',1:atomno);
x=xyz(:,1:3:end);
y=xyz(:,2:3:end);
z=xyz(:,3:3:end);

delx = x-repmat(mean(x),length(x),1);
dely = y-repmat(mean(y),length(y),1);
delz = z-repmat(mean(z),length(z),1);

deltaRs=sqrt([delx dely delz].^2);

binsall=[];
binno = 8;   % bin_num_opt=mean_fluctuation·(1+log2N)
for i=1:atomno;
    edges=[];
    diff=(max(deltaRs(:,i))-min(deltaRs(:,i)))/binno;
    for j=0:binno;
        edges=[edges min(deltaRs(:,i))+j*diff];
    end
    [Nth,edg,binb]=histcounts(deltaRs(:,i),edges);  %histcounts(X)
                                                    
    binsall = [binsall binb];  
end
	
[m,n]=size(binsall);
ns=[5];
tau=ns*50;  %This changes according to your stride when saving the coordinates. You have 100 ns and you saved 10000 frames. 10000 frames = 100 ns, 100 frames= 1 ns 
tau=fix(tau);
%Taucell= cell(length(tau),1); %We tried running it for several taus(1:5:20), thats why its a cell.
%for ind=1:length(tau);
    Tijtau=zeros(atomno,atomno);
    for i= 1:atomno;
        for j=1:atomno;
            tripleprobsijj= zeros(binno,binno,binno);
            tripleprobsjii= zeros(binno,binno,binno);
            for k =1:m-tau;%(ind);
                tripleprobsijj(binsall(k,i),binsall(k,j),binsall((tau+k),j))=tripleprobsijj(binsall(k,i),binsall(k,j),binsall((tau+k),j))+1;
                tripleprobsjii(binsall(k,j),binsall(k,i),binsall((tau+k),i))=tripleprobsjii(binsall(k,j),binsall(k,i),binsall((tau+k),i))+1;
            end
            pijj=tripleprobsijj./sum(sum(sum(tripleprobsijj)));
            pjii=tripleprobsjii./sum(sum(sum(tripleprobsjii)));
            hijj=[];
            hjii=[];
            for firstbin=1:binno;
                for secondbin=1:binno;
                    for thirdbin=1:binno;
                        if pijj(firstbin,secondbin,thirdbin)== 0;
                            pijj(firstbin,secondbin,thirdbin)=1;
                        end
                        if pjii(firstbin,secondbin,thirdbin)==0;
                            pjii(firstbin,secondbin,thirdbin)=1;
                        end
                        hijj= [hijj -pijj(firstbin,secondbin,thirdbin)*log(pijj(firstbin,secondbin,thirdbin))];
                        hjii= [hjii -pjii(firstbin,secondbin,thirdbin)*log(pjii(firstbin,secondbin,thirdbin))];
                    end
                end
            end
            Hijj= sum(hijj);
            Hjii= sum(hjii);
            Hjj= condEntropy(binsall((1:m-tau),j),binsall((tau+1):m,j));
            Hii= condEntropy(binsall((1:m-tau),i),binsall((tau+1):m,i));
            Hij = jointEntropy(binsall(:,i),binsall(:,j));
            Tijtau(i,j)=Hjj-Hijj+Hij;
            Tijtau(j,i)=Hii-Hjii+Hij;
        end
    end
%end
%    Taucell{ind}=Tijtau;
%Tijtaunonzeros100=Taucell{1};
Tijtaunonzeros=Tijtau;
Tijtaunonzeros(Tijtaunonzeros<0)=0; % Info cant be negative 
netTE = Tijtaunonzeros - Tijtaunonzeros'; 
norm_netTE=netTE./max(max(netTE)); 
norm_difference=sum(norm_netTE,2); 
