"""顺序轮转与正反轮次独立，偶数个版本也均匀覆盖每个执行位置。"""
def order(values,index,reverse=False):
    shift=index%len(values)
    result=values[shift:]+values[:shift]
    return result[::-1] if reverse else result
