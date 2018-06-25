/* 
 * File:   mRMR_test.h
 * Author: Skyrim
 *
 * Created on 2017年7月1日, 上午9:41
 */
#pragma once

#include <limits.h>

#include "incrementalLearner.h"
#include "distributionTree.h"
#include "xxyDist.h"
#include "xxxyDist.h"
#include "yDist.h"

class mRMR_test : public IncrementalLearner {
public:
    mRMR_test();
    mRMR_test(char*const*& argv, char*const* end);
    ~mRMR_test(void);

    void reset(InstanceStream &is); ///< reset the learner prior to training
    void initialisePass(); ///< must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)
    void train(const instance &inst); ///< primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass
    void finalisePass(); ///< must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)
    bool trainingIsFinished(); ///< true iff no more passes are required. updated by finalisePass()
    void getCapabilities(capabilities &c);

    virtual void classify(const instance &inst, std::vector<double> &classDist);


protected:
    unsigned int pass_; ///< the number of passes for the learner
    unsigned int k_; ///< the maximum number of parents
    unsigned int noCatAtts_; ///< the number of categorical attributes.
    unsigned int noClasses_; ///< the number of classes

    bool onlyAtt_;
    bool onlyK_; 

    std::vector<double> foldLossAttSelsect; 
    std::vector<double> foldLossforK; 
    std::vector<std::vector<double> > foldLossforAttandK;

    std::vector<CategoricalAttribute> order_;
    InstanceCount trainSize_;
    std::vector<bool> active_;
    unsigned int bestK_;

    xxyDist dist_; // used in the first pass
    yDist classDist_; // used in the second pass and for classification
    std::vector<distributionTree> dTree_; // used in the second pass and for classification
    std::vector<std::vector<CategoricalAttribute> > parents_;
    InstanceStream* instanceStream_;
};
