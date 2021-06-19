make
mpirun -n 4 ./BiotDD 
ls
make
mpirun -n 4 ./BiotDD 
ls
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
mpirun -n 16 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
rm CMakeCache.txt 
rm cmake_install.cmake 
rm Makefile 
cmake -DDEAL_II_DIR=/home/maj136/dealii_new/dealii-8.5.0/installed/
make release
ls
rm *vtu
ls
rm error4domains.tex 
ls
rm error16domains.tex 
rm CMakeCache.txt 
rm BiotDD 
rm -r CMakeFiles/
ls
rm cmake_install.cmake 
cmake -DDEAL_II_DIR=/home/maj136/dealii_new/dealii-8.5.0/installed/
make release
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
ls
rm *vtu
ls
make
mpirun -n 4 ./BiotDD 
make
ls
rm *vtu
ls

make
mpirun -n 4 ./BiotDD 
make
emacs solution_bar.txt 
emacs solution_bar_mortar.txt
make
mpirun -n 4 BiotDD 
mpirun -n 4 ./BiotDD 
emacs solution_bar_mortar.txt
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
make release
mpirun -n 16 ./BiotDD 
mpirun -n 4 ./BiotDD 
make release
mpirun -n 4 ./BiotDD 
mpirun -n 16 ./BiotDD 
make release
mpirun -n 16 ./BiotDD 
make release
mpirun -n 16 ./BiotDD 
make
mpirun -n 16 ./BiotDD 
cd BiotDD2/
ls
cd Example_3/
ls
cd HG_ML
ls
make release
make
mpirun -n 4 ./BiotDD 
mpirun -n 16 ./BiotDD 
ls
rm *.vtu
ls
exit
cd BiotDD2/
ls
cd Example_3/
ls
cd HG_ML/
ls
make
mpirun -n 16 ./BiotDD 
exit
ls
cd Example_2_new/
ls
cd 8D/
ls
emacs 8_output.txt 
cd ..
mkdir ~/arxiv
mkdir ~/arxiv/BiotDD1/
mkdir ~/arxiv/BiotDD1/Example_2
mkdir ~/arxiv/BiotDD1/Example_2/monolithic
emacs 8_output.txt 
ls
cd 8D/
ls
emacs 8_output.txt 
cp -r ../8D/. ~/arxiv/BiotDD1/Example_2/monolithic/
cd ..
mkdir ~/arxiv/BiotDD1/Example_2/DS
mkdir ~/arxiv/BiotDD1/Example_2/FS
ls
cd DS
ls
cd ..
cp -r DS/. ~/arxiv/BiotDD1/Example_2/DS/
cp -r FS/. ~/arxiv/BiotDD1/Example_2/FS/
cd ~/arxiv/BiotDD1/Example_2/FS/
cd ..
ls
cd monolithic/
ls
cd ..
cls
ls
cd ..
ls
cd ..
ls
cd ..
ls
cd BiotDD1/
ls
cd Monolithic/
ls
cd ..
ls
cd ..
ls
cd SplittingSchemes/
ls
cd FS
ls
cd ..
ls
cd ML
ls
cd ..
ls
cd ..
ls
cd BiotSplit/
ls
cd ..
ls
cd BiotDD1
ls
cd FS
ls
cd ..
ls
cd ..
s
ls
mkdir temp_run
ls
cd temp_run
ls
cd BiotDDMortar/
sls
ls
rm -r CMakeFiles/
rm cmake_install.cmake 
rm CMakeCache.txt 
rm Makefile 
cmake -DDEAL_II_DIR=/home/maj136/dealii_new/dealii-8.5.0/installed/
cd ..
ls
cd ..
ls
cd BiotSplit/
ls
cp inc/projector.h ../temp_run/BiotDDMortar/inc/
cd ..
cd temp_run/BiotDDMortar/
make release
rm *vtu
ls
rm *lyx
rm *tex
ls
nohup mpirun -n 4 ./BiotDD >>4_domain_linar.out &
emacs 4_domain_linar.out 
ls
emacs 4_domain_linar.out 
mpirun -n 4 BiotDD 
mpirun -n 4 ./BiotDD 
cd BiotDD2/Example_3/HG_ML/
cd .
cd ..
mkdir HG_DS
cp -r HG_ML/src/ HG_DS/
cp -r HG_ML/inc/ HG_DS/
cp -r HG_ML/CMakeLists.txt HG_DS/
cp -r HG_ML/poros.txt HG_DS/
cp -r HG_ML/permx.dat HG_DS/
cd HG_DS/
ls
cmake -DDEAL_II_DIR=/home/maj136/dealii_new/dealii-8.5.0/installed/
emacs src/biot_dd.cc
make release
mpirun -n 4 ./BiotDD 
emacs src/biot_mfedd.cc
make release
mpirun -n 4 ./BiotDD 
mpirun -n 16 ./BiotDD 
rm *vtu
mpirun -n 16 ./BiotDD 
cd ../HG_ML/
ls
cd ../HG_DS/
cd BiotDD2/
ls
cd Example_3/
LS
ls
cd HG_ML
ls
rm *vtu
make
mpirun -n 4 BiotDD 
mpirun -n 4 ./BiotDD 
pwd
rm *vtu
mpirun -n 16 ./BiotDD 
make release
mpirun -n 16 ./BiotDD 
make release
mpirun -n 16 ./BiotDD 
make release
mpirun -n 16 ./BiotDD 
make
mpirun -n 16 ./BiotDD 
make
mpirun -n 16 ./BiotDD 
mpirun -n 4 ./BiotDD 
ls
rm -r *vtu
mpirun -n 4 ./BiotDD 
exit
cd BiotDD2/Example_3/HG_ML/
ls
rm *vtu
ls
make release
mpirun -n 4 BiotDD 
mpirun -n 4 ./BiotDD 
ls
rm *vtu
cd ..
ls
rm -rf HG_DS
mkdir HG_DS
cp -r HG_ML/. HG_DS/
cd HG_DS/
ls
rm cmake_install.cmake 
rm CMakeCache.txt 
rm -rf CMakeFiles/
rm Makefile 
cmake -DDEAL_II_DIR=/home/maj136/dealii_new/dealii-8.5.0/installed/
make release
mpirun -n 16 BiotDD 
mpirun -n 16 ./BiotDD 
ls
rm *vtu
ls
nohup mpirun -n 16 ./BiotDD >>ds_log.out &
cd ../HG_ML/
ls
make release
nohup mpirun -n 16 ./BiotDD >>monolithic_log.out &
ls
emacs monolithic_log.out 
cd ..
ls
cd HG_DS
ls
emacs ds_log.out 
ls
emacs ds_log.out 
emacs ../HG_ML/monolithic_log.out 
emacs ds_log.out 
emacs ../HG_ML/monolithic_log.out 
cd ../HG_ML/
ls
rm *vtu
ls
rm monolithic_log.out 
make release
mpirun -n 16 BiotDD 
mpirun -n 16 ./BiotDD 
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
ls
rm*vtu
rm *vtu
ls
make
mpirun -n 4 ./BiotDD 
cd Biot2
cd BiotDD2/
cd Example_3/
ls
cd HG_ML/
make
nohup mpirun -n 4 ./BiotDD>>ml_log.out &
ls
emacs ml_log.out 
ls
mkdir dummy
cd dummy/
wget https://github.com/eldarkh/ElasticityMMMFE.git
ls
wget https://github.com/eldarkh/ElasticityMMMFE/archive/refs/heads/master.zip
unzip master.zip 
ls
cd ElasticityMMMFE-master/
ls
cmake -DDEAL_II_DIR=/home/maj136/dealii_new/dealii-8.5.0/installed/
make release
cp ../../../HG_ML/inc/projector.h inc/.
make release
ls
mpirun -n 4 ElastDD 
mpirun -n 4 ./ElastDD 
cp ~/elast_dd.cc src/.
make release
mpirun -n 4 ElastDD 
mpirun -n 4 ./ElastDD 
cp ~/elasticity_mfedd.cc src/.
make
nohup mpirun -n 4 ./ElastDD >>a_elast_log.out &
ls
cp a_elast_log.out ~/
cd BiotDD2
cd Example_3/HG_ML/
make release
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
emacs src/biot_dd.cc
make
emacs src/biot_dd.cc
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
cd BiotDD2/Example_3/HG_ML/
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
cd BiotDD2/Example_3/HG_ML/
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
cd BiotDD2/
ls
cd Example_3/
cd HG_ML/
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
cd BiotDD2/Example_3/HG_ML/
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 

make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
cd BiotDD2/
cd Example_3/HG_ML/
make
mpirun -n 4 BiotDD 
mpirun -n 4 ./BiotDD 
rm -rf BiotDDMortar/
exit
cd BiotDD2/Example_3/HG_ML/inc/
ls
cd ..
cd ~
ls
cd arxiv/
ls
BiotDD1/
ls
cd BiotDD1/Example_2/
cd ..
ls
pwd
ls
mv Example\ 1/ Example_1
ls
cd Example_1
ls
cd monolithic/
ls
cd c_0\=0.001_dt\=0.01/
ls
cmake -DDEAL_II_DIR=/home/maj136/dealii_new/dealii-8.5.0/installed/
make release
mpirun -n 4 BiotDD 
mpirun -n 4 ./BiotDD 
nohup mpirun -n 4 ./BiotDD >output_log.dat &
emacs output_log.dat 
cd ..
ls
cd dt\=0.001
cmake -DDEAL_II_DIR=/home/maj136/dealii_new/dealii-8.5.0/installed/
make release
nohup mpirun -n 4 ./BiotDD >output_log.dat &
cd ..
cd dt\=0.01
ls
cmake -DDEAL_II_DIR=/home/maj136/dealii_new/dealii-8.5.0/installed/
make release
nohup mpirun -n 4 ./BiotDD >output_log.dat &
cd ..
cd dt\=0.1/
ls
cmake -DDEAL_II_DIR=/home/maj136/dealii_new/dealii-8.5.0/installed/
make release
nohup mpirun -n 4 ./BiotDD >output_log.dat &
cd ..
ls
cd c_0\=0.001_dt\=0.01/
emacs output_log.dat 
ls
rm a_elast_log.out 
cd arxiv/
ls
cd BiotDD1/Example_
cd BiotDD1/
ls
rm -rf Example_1/
ls
cd ..
ls
exit
cd arxiv/BiotDD1/Example_1/
cd monolithic/
cd dt\=0.1/
emacs output_log.dat 
cd ..
ls
cd ..
ls
cd Example_
cd Example_1/
ls
pwd
ls
cd monolithic/
ls
cd dt\=0.01
emacs src/biot_dd.cc 
make
mpirun -n 4 BiotDD 
mpirun -n 4 ./BiotDD 
pwd
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
cp inc/projector.h ~/BiotDDMortar/inc/
ls
cd BiotDDMortar/
rm CMakeCache.txt 
rm cmake_install.cmake 
rm -r CMakeFiles/
rm Makefile 
cmake -DDEAL_II_DIR=/home/maj136/dealii_new/dealii-8.5.0/installed/
make release
mpirun -n 4 ./BiotDD 
exit
cd BiotDD2/Example_3/HG_ML/
make
ls
make
mpirun -n 4 BiotDD 
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
cd BiotDD2/Example_3/HG_ML/
make
mpirun -n 64 ./BiotDD 
cd BiotDD2/Example_3/HG_ML/
make release
mpirun -n 4 ./BiotDD 
make release
mpirun -n 4 BiotDD 
mpirun -n 4 ./BiotDD 
make release
mpirun -n 16 ./BiotDD 
make release
mpirun -n 16 ./BiotDD 
emacs src/biot_dd.cc
make release
mpirun -n 64 ./BiotDD 
make
mpirun -n 64 ./BiotDD 
cd BiotDD2/Example_3/HG_ML/
make 
make
mpirun -n 16 ./BiotDD 
cd ..
ls
cd ~
cd 16_subdomains/
cmake -DDEAL_II_DIR=/home/maj136/dealii_new/dealii-8.5.0/installed/
cmake -DDEAL_II_DIR=/home/maj136/dealii_new/dealii-8.5.0/installed/ .
ls
cd quadratic_dt\=0.0001_c_0\=1/
cmake -DDEAL_II_DIR=/home/maj136/dealii_new/dealii-8.5.0/installed/ .
make release
make
mpirun -n 16 ./BiotDD 
cd ~/BiotDD2/Example_3/HG_ML/
make
mpirun -n 4 ./BiotDD 
make
mpirun -n 4 ./BiotDD 
cd BiotDD2/Example_3/HG_ML/
make
mpirun -n 16 ./BiotDD 
make
mpirun -n 16 ./BiotDD 
top
cd Example_2_new/
cd ..
cd BiotDD2/Example_3/HG_ML/
make
mpirun -n 16 ./BiotDD 
make
mpirun -n 16 ./BiotDD 
make
nohup mpirun -n 16 ./BiotDD >quadratic_16_sub_1_over_16_H.txt &
emacs quadratic_16_sub_1_over_16_H.txt 
ls
cd Example_2_new/
ls
cd ~
cd 16_subdomains/
ls
cd quadratic_dt\=0.0001_c_0\=1/
ls
emacs src/biot_dd.cc
emacs src/biot_mfedd.cc 
emacs src/biot_dd.cc
emacs src/biot_mfedd.cc 
emacs src/biot_dd.cc
make release
nohup mpirun -n 64 ./BiotDD >64_subdoms_quadratic.txt &
ls
emacs 64_subdoms_quadratic.txt 
emacs src/biot_dd.cc
make
mpirun -n 64 ./BiotDD 
cd BiotDD2/Example_3/HG_ML/
emacs inc/data.h
make release
mpirun -n 64 ./BiotDD 
make
mpirun -n 16 ./BiotDD 
make
mpirun -n 16 ./BiotDD 
ls
rm *vtu
ls
emacs src/biot_dd.cc
make
nohup mpirun -n 64 ./BiotDD >64_subs_quad_mortar_k=1
ls
rm quadratic_16_sub_1_over_16_H.txt 
ls
rmsvd 64_subs_quad_mortar_k\=1 
emacs 64_subs_quad_mortar_k\=1 
top
rm 64_subs_quad_mortar_k\=1 
ls
nohup mpirun -n 64 ./BiotDD >64_subs_quad_mortar_k=1 &
ls
emacs 64_subs_quad_mortar_k\=1 
cd BiotDD2/Example_3/HG_ML/
ls
rm *vtu
ls
emacs 64_subs_quad_mortar_k\=1 
exit
cd BiotDD2/Example_3/HG_N
cd BiotDD2/Example_3/HG_ML/
ls
cd BiotDD2/Example_3/HG_ML/
make
mpirun -n 64 ./BiotDD 
make
mpirun -n 16 ./BiotDD 
top
cp poros.txt .
cp ~/permx.dat .
mpirun -n 16 ./BiotDD 
cd BiotDD2/Example_3/HG_ML/
ls
rm*vtu
rm *vtu
ls
make release
mpirun -n 16 ./BiotDD 
ls
make
mpirun -n 16 ./BiotDD 
make
mpirun -n 16 ./BiotDD 
make
mpirun -n 16 ./BiotDD 
exit
cd quadratic_dt\=0.0001_c_0\=0.001/
emacs src/biot_dd.cc
cp ~/BiotDD2/Example_3/HG_ML/inc/projector.h inc/
cmake -DDEAL_II_DIR=/home/maj136/dealii_new/dealii-8.5.0/installed/ .
make release
la
make release
nohup mpirun -n 4 ./BiotDD >output_log.txt
top
nohup mpirun -n 4 ./BiotDD >output_log.txt &
ls
emacs output_log.txt 
ls
cp Example_2_new/16D/inc/projector.h BiotDDMortar/inc/
cd BiotDDMortar/
ls
rm CMakeCache.txt 
rm -r CMakeFiles/
rm Makefile 
cmake -DDEAL_II_DIR=/home/maj136/dealii_new/dealii-8.5.0/installed/ .
make release
mpirun -n 15 ./BiotDD 
pwd
emacs src/biot_mfedd.cc 
make
mpirun -n 15 ./BiotDD 
rm .
rm *
ls
rm -r .
ls
rm -r CMakeFiles/
rfm -r CMakeFiles/
rm -rf CMakeFiles/
rm -rf inc/
rm -rf src
ls
rm *vtu
ls
rm -r CMakeFiles/
rm CMakeCache.txt 
rm Makefile 
cp ../BiotDD2/Example_3/HG_ML/inc/projector.h inc/
cmake -DDEAL_II_DIR=/home/maj136/dealii_new/dealii-8.5.0/installed/ .
make release
mpirun -n 15 ./BiotDD 
emacs src/biot_dd.cc 
make
mpirun -n 15 ./BiotDD 
make release
mpirun -n 15 ./BiotDD 
emacs src/biot_mfedd.cc 
make
mpirun -n 15 ./BiotDD 
make
mpirun -n 15 ./BiotDD 
emacs src/biot_mfedd.cc 
make
mpirun -n 15 ./BiotDD 
cd ..
ls
rm -r quadratic_dt\=0.0001_c_0\=0.001/
rm -rf quadratic_dt\=0.0001_c_0\=0.001/
rm -rf BiotDDMortar/
cd BiotDD2/Example_3/HG_ML/
emacs inc/utilities.h 

make release
make
ls
rm *vtu
ls
cmake -DDEAL_II_DIR=/home/maj136/dealii_new/dealii-8.5.0/installed/ .
cd ../HG_DS/
make
cd ..
ls
cd HG_ML/
ls
cd ..
rm -rf HG_ML
mkdir HG_ML
ls
cd HG_ML
ls
mkdir inc
ls
cd inc/
ls
cd ..
ls
cp -r ../HG_DS/inc/. inc/
cd inc
ls
cp ~/utilities.h .
ls
rm biot_mfedd.h~
ls
cd ..
ls
cp ../HG_DS/CMakeLists.txt .
cp ~/poros.txt 
cp ~/poros.txt .
cp ~/permx.dat .
ls
cmake -DDEAL_II_DIR=/home/maj136/dealii_new/dealii-8.5.0/installed/ .
make release
mpirun -n 15 ./BiotDD 
make
mpirun -n 15 ./BiotDD 
make
mpirun -n 15 ./BiotDD 
make
mpirun -n 15 ./BiotDD 
make
mpirun -n 15 ./BiotDD 
make
mpirun -n 15 ./BiotDD 
make
mpirun -n 15 ./BiotDD 
make
mpirun -n 15 ./BiotDD 
make
mpirun -n 15 ./BiotDD 
make
mpirun -n 15 ./BiotDD 
emacs src/biot_dd.cc 
make
mpirun -n 15 ./BiotDD 
make
mpirun -n 15 ./BiotDD 
emacs src/biot_dd.cc
make
emacs src/biot_dd.cc
make
emacs src/biot_dd.cc
make
mpirun -n 15 ./BiotDD 
make
mpirun -n 15 ./BiotDD 
make
mpirun -n 15 ./BiotDD 
make
mpirun -n 15 ./BiotDD 
make
mpirun -n 15 ./BiotDD 
make
mpirun -n 15 ./BiotDD 
rm *vtu
make
mpirun -n 15 ./BiotDD 
cd BiotDD2/Example_3/HG_ML/
cp ~/utilities.h inc/
make
mpirun -n 15 ./BiotDD 
emacs interace_marks.txt 
rm interace_marks.txt 
make
mpirun -n 15 ./BiotDD 
make
mpirun -n 15 ./BiotDD 
make
mpirun -n 15 ./BiotDD 
make
mpirun -n 15 ./BiotDD 
rm interace_marks.txt 
make
mpirun -n 15 ./BiotDD 
pwd
mpirun -n 15 ./BiotDD 
cp ~/utilities.h inc/
rm interace_marks.txt 
make 
mpirun -n 15 ./BiotDD 
make 
mpirun -n 15 ./BiotDD 
rm *vtu
make
mpirun -n 15 ./BiotDD 
rm *vtu
make
mpirun -n 15 ./BiotDD 
make
rm *vtu
mpirun -n 15 ./BiotDD 
cd BiotDD2/Example_3/HG_ML/
make 
mpirun -n 15 ./BiotDD 
cd BiotDD2/Example_3/HG_ML/
make 
rm *vtu
nohup mpirun -n 15 ./BiotDD >Example_2_fine_scale.out &
ls
emacs Example_2_fine_scale.out 
cd ~
ls
cp -r BiotDD-Heterogeneious/. BiotDD-Heterogeneous/
rm -rf BiotDD-Heterogeneious/
cd BiotDD-Heterogeneous/
ls
cp ../poros.txt .
ls
cd ..
cp BiotDD2/Example_3/HG_ML/inc/projector.h BiotDD-Heterogeneous/inc/
cd BiotDD-Heterogeneous/
cmake -DDEAL_II_DIR=/home/maj136/dealii_new/dealii-8.5.0/installed/ .
make release
mpirun -n 15 ./BiotDD 
ls
make
rm -r ../BiotDD-Heterogeneious/
make
mpirun -n 15 ./BiotDD 
pwd
cd ..
rm -rf BiotDD-Heterogeneous/
cd BiotDD2/Example_3/HG_ML/
emacs Example_2_fine_scale.out 
cd ~/dt\=0.001_quadratic_one_per_cell/
cp ../BiotDD2/Example_3/HG_ML/inc/projector.h inc/
cmake -DDEAL_II_DIR=/home/maj136/dealii_new/dealii-8.5.0/installed/ .
make release
emacs src/biot_mfedd.cc 
make
nohup mpirun -n 15 ./BiotDD >Example_2_dt=0.001_quad_one.out &
ls
emacs Example_2_dt\=0.001_quad_one.out 
cp BiotDD2/Example_3/HG_ML/inc/projector.h .
cd linear_one_per_interface/
cp ../projector.h inc/
cmake -DDEAL_II_DIR=/home/maj136/dealii_new/dealii-8.5.0/installed/ .
make release
mpirun -n 15 ./BiotDD 
emacs src/biot_dd.cc 
make
mpirun -n 15 ./BiotDD 
cd BiotDD2/Example_3/HG_ML/
ls
pwd
exit
