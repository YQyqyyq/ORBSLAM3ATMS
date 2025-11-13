#ifndef EDGE_SIM3_PRIOR_H
#define EDGE_SIM3_PRIOR_H

#include <Thirdparty/g2o/g2o/core/base_unary_edge.h>
#include <Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h> // for Sim3 and VertexSim3Expmap

namespace g2o {

class EdgeSim3Prior : public BaseUnaryEdge<7, Sim3, VertexSim3Expmap> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeSim3Prior() {}

    // 计算误差：log( measurement^{-1} * estimate )
    void computeError() override {
        const VertexSim3Expmap* v = static_cast<const VertexSim3Expmap*>(_vertices[0]);
        Sim3 est = v->estimate();
        // 误差 = 期望位姿的逆 * 估计位姿，然后取对数映射到李代数空间
        Sim3 d = _measurement.inverse() * est;
        _error = d.log();
    }

    // g2o的Sim3对数映射已经处理了尺度，所以我们不需要特殊处理
    // The error is a 7-vector, where the first 3 components are rotation,
    // the next 3 are translation, and the last one is log(scale).

    // 不需要文件 I/O，返回 false
    bool read(std::istream& )  override { return false; }
    bool write(std::ostream& ) const override { return false; }
};

} // namespace g2o

#endif // EDGE_SIM3_PRIOR_H