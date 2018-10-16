import ai_f18_hw06 as ai

lsvm = 'lin_svm.pck'
rsvm = 'rbf_svm.pck'
p2svm = 'poly_2_svm.pck'
p3svm = 'poly_3_svm.pck'
p4svm = 'poly_4_svm.pck'
p5svm = 'poly_5_svm.pck'

ai.train_and_persist(ai.lin_svm, ai.mnist_train_data_dc, ai.mnist_train_target_dc, lsvm)
ai.train_and_persist(ai.rbf_svm, ai.mnist_train_data_dc, ai.mnist_train_target_dc, rsvm)
ai.train_and_persist(ai.poly_svm_2, ai.mnist_train_data_dc, ai.mnist_train_target_dc, p2svm)
ai.train_and_persist(ai.poly_svm_3, ai.mnist_train_data_dc, ai.mnist_train_target_dc, p3svm)
ai.train_and_persist(ai.poly_svm_4, ai.mnist_train_data_dc, ai.mnist_train_target_dc, p4svm)
ai.train_and_persist(ai.poly_svm_5, ai.mnist_train_data_dc, ai.mnist_train_target_dc, p5svm)

ai.print_svm_report(lsvm, ai.mnist_test_data_dc, ai.mnist_test_target_dc)
ai.print_svm_report(rsvm, ai.mnist_test_data_dc, ai.mnist_test_target_dc)
ai.print_svm_report(p2svm, ai.mnist_test_data_dc, ai.mnist_test_target_dc)
ai.print_svm_report(p3svm, ai.mnist_test_data_dc, ai.mnist_test_target_dc)
ai.print_svm_report(p4svm, ai.mnist_test_data_dc, ai.mnist_test_target_dc)
ai.print_svm_report(p5svm, ai.mnist_test_data_dc, ai.mnist_test_target_dc)