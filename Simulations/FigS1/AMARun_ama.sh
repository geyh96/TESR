


cd ./Case_TESR
bash ARrun.sh
bash ARrun1.sh
cd ..


cd ./Case_DDR
bash ARrun.sh
bash ARrun1.sh
cd ..


cd ./Case_TransIRM
bash ARrun.sh
bash ARrun1.sh
cd ..




cd ./Case_DNN
bash ARrun.sh
bash ARrun1.sh
cd ..


cd ./Case_FT
bash ARrun.sh
bash ARrun1.sh
cd ..

wait

cd /home/22118785r/TESR_github/Simulations/Table1

bash AMARun_ama.sh

wait

cd /home/22118785r/TESR_github/Simulations/FigS2

bash AMARun_ama.sh

wait

cd /home/22118785r/TESR_github/Simulations/FigS3/FigS3.1

bash AMARun_ama.sh

wait

cd /home/22118785r/TESR_github/Simulations/FigS3/FigS3.2

bash AMARun_ama.sh

wait

cd /home/22118785r/TESR_github/Simulations/FigS4

bash AMARun_ama.sh

wait

cd /home/22118785r/TESR_github/Simulations/FigS2

bash AMARun_ama.sh

wait
# find . -name "*.pt" -print0 | xargs -0 rm

# find . -name "*.log" -print0 | xargs -0 rm
