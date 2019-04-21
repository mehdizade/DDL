for i in {2..15}
do
	cp  "lasso_fista_Layer1.cc"   "lasso_fista_Layer"$i".cc"
        cp  "lasso_fista_Layer1.cu.cc"   "lasso_fista_Layer"$i".cu.cc"
	echo $i
done
