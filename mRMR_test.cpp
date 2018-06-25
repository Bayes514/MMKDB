/* 
 * File:   mRMR_test.cpp
 * Author: Liu Yang
 * 
 * Created on 2017年6月27日, 上午10:30
 */
#include <assert.h>
#include <math.h>
#include <set>
#include <algorithm>
#include <stdlib.h>

#include "mRMR_test.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"

mRMR_test::mRMR_test() : pass_(1) {
}

mRMR_test::mRMR_test(char*const*& argv, char*const* end) : pass_(1) {
    name_ = "mRMR_test";

    // defaults
    k_ = 1;
    onlyAtt_ = false;
    onlyK_ = false;

    // get arguments
    while (argv != end) {
        if (*argv[0] != '-') {
            break;
        } else if (argv[0][1] == 'k') {
            getUIntFromStr(argv[0] + 2, k_, "k");
        } else if (streq(argv[0] + 1, "onlyAtt")) {
            printf("onlyAtt\n");
            onlyAtt_ = true;
        } else if (streq(argv[0] + 1, "onlyK")) {
            printf("onlyK\n");
            onlyK_ = true;
        } else {
            break;
        }

        name_ += argv[0];

        ++argv;
    }
}

mRMR_test::~mRMR_test(void) {
}

void mRMR_test::getCapabilities(capabilities &c) {
    c.setCatAtts(true); // only categorical attributes are supported at the moment
}

// creates a comparator for two attributes based on their relative mutual information with the class

class miCmpClass {
public:

    miCmpClass(std::vector<float> *m) {
        mi = m;
    }

    bool operator()(CategoricalAttribute a, CategoricalAttribute b) {
        return (*mi)[a] > (*mi)[b];
    }

private:
    std::vector<float> *mi;
};

void mRMR_test::reset(InstanceStream &is) {
    instanceStream_ = &is;
    const unsigned int noCatAtts = is.getNoCatAtts();
    noCatAtts_ = noCatAtts;
    noClasses_ = is.getNoClasses();

    k_ = min(k_, noCatAtts_ - 1); // k cannot exceed the real number of categorical attributes - 1
    // initialise distributions
    dTree_.resize(noCatAtts);
    parents_.resize(noCatAtts);

    for (CategoricalAttribute a = 0; a < noCatAtts; a++) {
        parents_[a].clear();
        dTree_[a].init(is, a);
    }

    /*初始化各数据结构空间*/
    dist_.reset(is); //
    classDist_.reset(is);

    pass_ = 1;

    //***********************************
    order_.clear();
    trainSize_ = 0;
    active_.assign(noCatAtts_, false);

    if (onlyAtt_)
        foldLossAttSelsect.assign(noCatAtts_ + 1, 0.0);
    else if (onlyK_)
        foldLossforK.resize(k_ + 1);
    else {
        //k值和特征子集自适应
        foldLossforAttandK.resize(k_ + 1); 
        for (int i = 0; i <= k_; i++) {
            foldLossforAttandK[i].assign(noCatAtts_ + 1, 0);
        }
    }
}

/// primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass
/*通过训练集来填写数据空间*/
void mRMR_test::train(const instance &inst) {
    if (pass_ == 1) {
        dist_.update(inst);
        trainSize_++; // to calculate the RMSE for each LOOCV
    } else if (pass_ == 2) {
        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            dTree_[a].update(inst, a, parents_[a]);
        }
        classDist_.update(inst);
    } else {
        assert(pass_ == 3);
        if (onlyAtt_) {     //只做特征选择
            std::vector<double> posteriorDist(noClasses_);
            //Only the class is considered
            for (CatValue y = 0; y < noClasses_; y++) {
                posteriorDist[y] = classDist_.ploocv(y, inst.getClass()); //Discounting inst from counts
            }
            normalise(posteriorDist);
            const CatValue trueClass = inst.getClass();
            const double error = 1.0 - posteriorDist[trueClass];

            foldLossAttSelsect[noCatAtts_] += error*error;

            for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin(); it != order_.end(); it++) {
                dTree_[*it].updateClassDistributionloocv(posteriorDist, *it, inst); //Discounting inst from counts
                normalise(posteriorDist);
                const double error = 1.0 - posteriorDist[trueClass];

                foldLossAttSelsect[*it] += error*error;

            }   
        } else if (onlyK_) {    //只做K值选择
            std::vector<std::vector<double> > posteriorDist(k_ + 1); //+1 for NB (k=0)
            for (int k = 0; k <= k_; k++) {
                posteriorDist[k].assign(noClasses_, 0.0);
            }
            //Only the class is considered
            for (CatValue y = 0; y < noClasses_; y++) {
                posteriorDist[0][y] = classDist_.ploocv(y, inst.getClass()); //Discounting inst from counts
            }
            for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin();
                    it != order_.end(); it++) {
                dTree_[*it].updateClassDistributionloocvWithNB(posteriorDist, *it, inst, k_); //Discounting inst from counts           
                for (int k = 0; k <= k_; k++) {
                    normalise(posteriorDist[k]);
                }
            }
            const CatValue trueClass = inst.getClass();
            for (int k = 0; k <= k_; k++) {
                normalise(posteriorDist[k]);
                const double error = 1.0 - posteriorDist[k][trueClass];
                foldLossforK[k] += error*error;
            }
        } else {    //特征选择 + K值选择
            std::vector<std::vector<double> > posteriorDist(k_ + 1); //+1 for NB (k=0)
            for (int k = 0; k < k_ + 1; k++) {
                posteriorDist[k].assign(noClasses_, 0.0);
            }
            //初始化后验概率
            for (CatValue y = 0; y < noClasses_; y++) {
                posteriorDist[0][y] = classDist_.ploocv(y, inst.getClass()); //Discounting inst from counts
            }
            normalise(posteriorDist[0]);

            const CatValue trueClass = inst.getClass();
            const double error = 1.0 - posteriorDist[0][trueClass];
            foldLossforAttandK[0][noCatAtts_] += error*error;

            //计算RMSE二维矩阵
            for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin();
                    it != order_.end(); it++) {
                dTree_[*it].updateClassDistributionloocvWithNB(posteriorDist, *it, inst, k_); //Discounting inst from counts
                for (int k = 0; k <= k_; k++) {
                    normalise(posteriorDist[k]);
                    const double error = 1.0 - posteriorDist[k][trueClass];
                    foldLossforAttandK[k][*it] += error*error;
                }
            }
           
        }
    }
}

/// must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)

void mRMR_test::initialisePass() {
}

/// must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)

void mRMR_test::finalisePass() {
    if (pass_ == 1) {
        // calculate the mutual information from the xy distribution
        std::vector<float> mi;
        getMutualInformation(dist_.xyCounts, mi);

        // calculate the conditional mutual information from the xxy distribution
        crosstab<float> cmi = crosstab<float>(noCatAtts_);
        getCondMutualInf(dist_, cmi);

        // 初始化属性次序
        std::vector<CategoricalAttribute> order_mi;

        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            order_mi.push_back(a);
        }

        // assign the parents
        if (!order_mi.empty()) {
            miCmpClass cmp(&mi);

            std::sort(order_mi.begin(), order_mi.end(), cmp);

            /***************************** mRMR Ranking*********************************************/
            crosstab<float> mi_xixj = crosstab<float>(noCatAtts_); 
            getMutualInformation_XiXj(dist_, mi_xixj);

            std::vector<CategoricalAttribute> order_mRMR;
            order_mRMR.clear();
            mRMR(dist_, mi, order_mi, mi_xixj, order_mRMR, MID);
            order_mi.clear();
        
            for (CategoricalAttribute a = 0; a < order_mRMR.size(); a++) {
                order_.push_back(order_mRMR[a]);
                active_[order_mRMR[a]] = true;
            }

            dist_.clear();
            /***************************************对照实验mkdb***********************************/
//             for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
//                order_.push_back(order_mi[a]);
//                 active_[order_mi[a]] = true;
//            }
            /**************************************************************************/
            // proper mRMR_test assignment of parents
            for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin() + 1; it != order_.end(); it++) {
                parents_[*it].push_back(order_[0]);
                for (std::vector<CategoricalAttribute>::const_iterator it2 = order_.begin() + 1; it2 != it; it2++) {
                    // make parents into the top k attributes on mi that precede *it in order
                    if (parents_[*it].size() < k_) {
                        // create space for another parent
                        // set it initially to the new parent.
                        // if there is a lower value parent, the new parent will be inserted earlier and this value will get overwritten
                        parents_[*it].push_back(*it2);
                    }
                    for (unsigned int i = 0; i < parents_[*it].size(); i++) {
                        if (cmi[*it2][*it] > cmi[parents_[*it][i]][*it]) {
                            // move lower value parents down in order
                            for (unsigned int j = parents_[*it].size() - 1; j > i; j--) {
                                parents_[*it][j] = parents_[*it][j - 1];
                            }
                            // insert the new att
                            parents_[*it][i] = *it2;
                            break;
                        }
                    }
                }

            }
        }
    } else if (pass_ == 3) {
        std::vector<CategoricalAttribute>::const_iterator bestattIt = order_.end() - 1;
        bestK_ = 0;

        if (onlyAtt_) {//特征选择
            for (int i = 0; i < order_.size(); i++) {
                foldLossAttSelsect[i] = sqrt(foldLossAttSelsect[i] / trainSize_);
            }
            double globalmin = foldLossAttSelsect[order_.size() - 1];
            for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin(); it != order_.end(); it++) {
                if (foldLossAttSelsect[*it] < globalmin) {
                    globalmin = foldLossAttSelsect[*it];
                    bestattIt = it; //order_序列中的最佳特征子集0~bestNumofAtt_
                }
            }
            
            for (std::vector<CategoricalAttribute>::const_iterator it = bestattIt + 1; it != order_.end(); it++) {
                active_[*it] = false;
            }

        } else if (onlyK_) {//K值选择
            //foldLossforK
            for (unsigned int k = 0; k <= k_; k++) {
                foldLossforK[k] = sqrt(foldLossforK[k] / trainSize_);
            }
            double globalmin = foldLossforK[0];
            for (unsigned int k = 0; k <= k_; k++) {
                if (foldLossforK[k] < globalmin) {
                    globalmin = foldLossforK[k];
                    bestK_ = k;
                }
            }
     
        } else {//特征选择 + K值选择
            //foldLossforAttandK  
            for (unsigned int k = 0; k <= k_; k++) {
                for (unsigned int att = 0; att < noCatAtts_ + 1; att++) {
                    foldLossforAttandK[k][att] = sqrt(foldLossforAttandK[k][att] / trainSize_);                                     
                }               
            }

            double globalmin = foldLossforAttandK[0][noCatAtts_];
            for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin(); it != order_.end(); it++) {
                for (unsigned int k = 0; k <= k_; k++) {
                    if (foldLossforAttandK[k][*it] < globalmin) {
                        globalmin = foldLossforAttandK[k][*it];
                        bestattIt = it;
                        bestK_ = k;
                    }
                }
            }
           
            for (std::vector<CategoricalAttribute>::const_iterator it = bestattIt + 1; it != order_.end(); it++) {
                active_[*it] = false;
            }
        }
    } else {
        assert(pass_ == 2);
    }
    ++pass_;
}

/// true iff no more passes are required. updated by finalisePass()
bool mRMR_test::trainingIsFinished() {
    return pass_ > 3;
}

void mRMR_test::classify(const instance& inst, std::vector<double> &posteriorDist) {
   
    // calculate the class probabilities in parallel
    // P(y)
    for (CatValue y = 0; y < noClasses_; y++) {
        posteriorDist[y] = classDist_.p(y) * (std::numeric_limits<double>::max() / 2.0); // scale up by maximum possible factor to reduce risk of numeric underflow
    }
    // P(x_i | x_p1, .. x_pk, y)
    for (CategoricalAttribute x = 0; x < noCatAtts_; x++) {
        if (active_[x]) {
            if (onlyAtt_)
                dTree_[x].updateClassDistribution(posteriorDist, x, inst);
            else {
                // AttandK || onlyK_
                dTree_[x].updateClassDistributionForK(posteriorDist, x, inst, bestK_);
            }
        }
    }
    // normalise the results
    normalise(posteriorDist);
}


