import tensorflow as tf

def din_deepfm_mmoe(features, mode, params):
    feature_indices, feature_values = features['feature_indices'], features['feature_values']
    #[B,1]
    #获取batch_size内省份的编码信息
    domain_indices = tf.gather(params=feature_indices, indices=params['index.province'], axis=1)
    task_output,ls_loss = mmoe_modules(features, mode, params)
    #net = task_output
    task_list = params['network.task_list']    #['sn','gd','gx','jx']
    final_task = []
    for task_idx in range(len(task_list)):
        final_task.append(task_output["task" + str(task_idx)])
    #通过task对应的Gate来得到多个Expert的加权平均，然后输入到task对应的Tower层（MLP网络层）;
    #domain_indices [B,1]
    #[B]
    domain_indices = tf.reshape(domain_indices, [tf.shape(domain_indices)[0]])
    depth = len(task_list)
    #[B,len]
    pro_max = tf.one_hot(domain_indices, depth,dtype=tf.float32)
    #[B,1,len]
    pro_max = tf.expand_dims(pro_max, axis=1)

    #ex [0,1,2,0,1,2]
    #[[1., 0., 0.],
    #  [0., 1., 0.],
    #  [0., 0., 1.],
    #  [1., 0., 0.],
    #  [0., 1., 0.],
    #  [0., 0., 1.]]
    #[B,len,W]
    final_result = tf.concat([task for task in final_task], axis=1)
    #[B,1,W]
    net = tf.matmul(pro_max,final_result)
    #[B,W]
    net = tf.squeeze(net, axis=1)

    for idx, units in enumerate([256, 32]):
        net = tf.compat.v1.layers.dense(inputs=net, units=units,
                                            kernel_initializer=tf.compat.v1.keras.initializers.he_normal(),
                                            kernel_regularizer=tf.compat.v1.keras.regularizers.l2(0.01),
                                            name='deepfm_din_dense_layer_{}'.format(idx))
        net = tf.nn.relu(net, name='deepfm_din_relu_layer_{}'.format(idx))

        #[B,1]
    net = tf.compat.v1.layers.dense(inputs=net, units=1,
                                        kernel_initializer=tf.compat.v1.keras.initializers.he_normal(),
                                        kernel_regularizer=tf.compat.v1.keras.regularizers.l2(0.01),
                                        name='deepfm_din_dense_layer_final')

    output = tf.sigmoid(net, name='out')


    return output,ls_loss




def din_deepfm_expert(features, mode, params):

    # ============= input =============
    # (feature_indices = feature_values = [batch_size, field_size], labels = [batch_size, 1])
    #[B,D']
    feature,feature_embed = combine_feature(features, mode, params)

    return feature,feature_embed


def cv_squared(x):
    """The squared coefficient of variation of a sample.
    Useful as a loss to encourage a positive distribution to be more uniform.
    Epsilons added for numerical stability.
    Returns 0 for an empty Tensor.
    Args:
    x: a `Tensor`.
    Returns:
    a `Scalar`.
    """
    eps = 1e-10
    mean, variance = tf.nn.moments(x, axes=0)
    return variance / (mean**2 + eps)

def build_mask_matrix(seqlen, bsz):
    res_list = []

    base_mask = tf.ones((seqlen,seqlen),dtype = tf.float32) - tf.eye(seqlen)
    #bsz = len(valid_len_list)
    for i in range(bsz):
        one_base_mask = tf.Variable(base_mask)#[4,4]
        #print(one_base_mask)
        #one_valid_len = valid_len_list[i]

        #for j in range(one_valid_len,seqlen):
            #for k in range(seqlen):
                #print(j)
                #one_base_mask[k,j].assign(0.) # = 0.
                #one_base_mask[j,k].assign(0.) # = 0.
        res_list.append(one_base_mask)

    res_mask = tf.stack(res_list,0)
    #print(res_mask)
    #assert res_mask.shape == tf.shape([bsz, seqlen, seqlen])
    return tf.convert_to_tensor(res_mask)

def contrastive_loss(score_matrix):
    '''
       score_matrix: bsz x seqlen x seqlen
       input_ids: bsz x seqlen
    '''
    bsz, seqlen, _ = score_matrix.shape  #.size()
    #获取对角线的数据
    gold_score = tf.linalg.diag_part(score_matrix,k=0) # bsz x seqlen
    gold_score = tf.expand_dims(gold_score,-1)#bsz x seqlen x 1
    #assert gold_score.shape == [bsz, seqlen, 1]
    difference_matrix = gold_score - score_matrix
    #assert difference_matrix.shape == [bsz, seqlen, seqlen]
    margin = 0.3
    loss_matrix = margin - difference_matrix # bsz x seqlen x seqlen
    loss_matrix = tf.nn.relu(loss_matrix) #torch.nn.functional.relu(loss_matrix)
    #pad_token_id = 0.0
    ### input mask
    #input_mask = tf.ones_like(input_ids,dtype = tf.float32) #torch.ones_like(input_ids).type(torch.FloatTensor)
    #input_mask = tf.where(input_ids!=pad_token_id,input_mask,0.0)#input_mask.masked_fill(input_ids.eq(self.pad_token_id),0.0)
    #valid_len_list = tf.reduce_sum(input_mask,axis = -1)
    #valid_len_list = list(valid_len_list.numpy())
    #loss_mask = build_mask_matrix(seqlen,bsz)
    loss_mask = tf.ones_like(score_matrix,dtype=tf.float32) - tf.eye(seqlen)

    #if score_matrix.is_cuda:
    #    loss_mask = loss_mask.cuda(score_matrix.get_device())
    #将对角线内容置为0
    masked_loss_matrix = loss_matrix * loss_mask

    loss_matrix = tf.reduce_sum(masked_loss_matrix, axis = -1)
    #assert loss_matrix.shape == input_ids.shape
    #loss_matrix = loss_matrix * input_mask
    cl_loss = tf.reduce_sum(loss_matrix) / tf.reduce_sum(loss_mask)
    return cl_loss

def mmoe_modules(features,mode,params):
    #features [B,D']
    task_list = params['network.task_list']    #['sn','gd','gx','jx']
    expert_num = params['network.expert_num']  # 8
    expert_hidden_units = params['network.expert_hidden_units']  # [256, 32]

    expert_inputs = {}
    #专家网络输入都是相同的input
    for idx in range(expert_num):
        expert_inputs["expert" + str(idx)] = features
    #tf.compat.v1.name_scope
    #（1）在某个tf.compat.v1.name_scope()指定的区域中定义的所有对象及各种操作，他们的“name”属性上会增加该命名区的区域名，用以区别对象属于哪个区域；
    #（2）将不同的对象及操作放在由tf.name_scope()指定的区域中，便于在tensorboard中展示清晰的逻辑关系图，这点在复杂关系图中特别重要。
    #专家网络结构
    #模型输入会通过映射到，所有task共享的多个Expert, 一个Expert就是RELU激活函数的全连接层，称为Mixture-of-Experts
    expert_outputs = {}
    with tf.compat.v1.variable_scope("experts", reuse=tf.compat.v1.AUTO_REUSE):
        for expert_idx in range(expert_num):
            y = expert_inputs["expert" + str(expert_idx)]
            '''
            for idx, layer_size in enumerate(expert_hidden_units):
                scope_name_tmp = "expert_%d_layer_%d" % (expert_idx, idx)
                y = tf.compat.v1.layers.dense(inputs=y, units=layer_size,
                                                 kernel_initializer=tf.compat.v1.keras.initializers.he_normal(),
                                                 kernel_regularizer=tf.compat.v1.keras.regularizers.l2(0.01),
                                                 name=scope_name_tmp)
                y = tf.nn.relu(y, name="expert_%d_layer_%d_relu" % (expert_idx, idx))
            '''
            #激活函数？？？
            y,feature_embed = din_deepfm_expert(features, mode, params)
            y_new = tf.identity(y, name="expert_out"+str(expert_idx))
            #[B,1,w]
            y = tf.expand_dims(y_new, axis=1)
            expert_outputs["expert" + str(expert_idx)] = y
    #门控网络结构
    #模型输入还会映射到多个Gate，一个task独立拥有一个Gate，论文中Gate就是一个没有bias和激活函数的全连接层，然后接softmax，称为Multi-gate
    gates_output = {}
    importance_sum = tf.zeros([expert_num], tf.float32)
    cl_loss_sum = 0.0
    loss_coef = 1e-2
    with tf.compat.v1.variable_scope("gates_expert", reuse=tf.compat.v1.AUTO_REUSE):
        for task_idx in range(len(task_list)):
            scope_name_tmp = "gates_task_%d" % (task_idx)
            y = feature_embed
            z = tf.compat.v1.layers.dense(inputs=y, units=expert_num,
                                         kernel_initializer=tf.compat.v1.keras.initializers.he_normal(),
                                         kernel_regularizer=tf.compat.v1.keras.regularizers.l2(0.01),
                                         name=scope_name_tmp)
            #[B,expert_num]
            gate_out = tf.nn.softmax(z)
            # calculate importance loss
            #importance = gate_out.sum(0)
            importance = tf.reduce_sum(gate_out, 0)
            importance_sum += importance
            #
            #loss = cv_squared(importance)
            #loss *= loss_coef
            #loss_gates += loss
            # whether use dropout need analysis
            #gate_out = tf.compat.v1.layers.dropout(inputs=gate_out, rate=params['network.dropout_rate'], training=(mode == tf.estimator.ModeKeys.TRAIN),name='gate_dropout_layer_{}'.format(task_idx))
            #re-normalizing
            #gate_out = gate_out / (tf.reduce_sum(input_tensor=gate_out, axis=-1, keepdims=True) + 1e-6)

            gates_output["task" + str(task_idx)] = gate_out
    loss_gates = cv_squared(importance_sum)
    loss_gates *= loss_coef
    #增加图神经网络结
    all_gates = []
    for task_idx in range(len(task_list)):
        all_gates.append(gates_output["task" + str(task_idx)])
    #[task_num,B*expert_num]
    #all_gates_tensor = tf.reshape(all_gates,shape=[len(task_list),gates_output["task0"].shape[0]*gates_output["task0"].shape[1]])
    all_gates_tensor = tf.reshape(all_gates,shape=[len(task_list),-1])
    #构建相似度矩阵
    #[task_num,task_num]
    A = tf.constant([[1.0,0.966,0.724,0.838,0.789,0.781,0.826,0.825,0.815,0.937],
                   [0.966,1.0,0.723,0.839,0.759,0.758,0.862,0.785,0.839,0.936],
                   [0.724,0.723,1.0,0.879,0.796,0.764,0.625,0.842,0.563,0.619],
                   [0.838,0.839,0.879,1.0,0.901,0.883,0.744,0.939,0.729,0.767],
                   [0.789,0.759,0.796,0.901,1.0,0.995,0.706,0.880,0.703,0.717],
                   [0.781,0.758,0.764,0.883,0.995,1.0,0.700,0.849,0.711,0.721],
                   [0.826,0.862,0.625,0.744,0.706,0.700,1.0,0.727,0.845,0.830],
                   [0.825,0.785,0.842,0.939,0.880,0.849,0.727,1.0,0.734,0.752],
                   [0.815,0.839,0.563,0.729,0.703,0.711,0.845,0.734,1.0,0.834],
                   [0.937,0.936,0.619,0.767,0.717,0.721,0.830,0.752,0.834,1.0]])
    B = tf.zeros([len(task_list),16])
    Adj = A + tf.matmul(B,tf.transpose(B))
    #[task_num,task_num]
    Adj = tf.nn.softmax(Adj,axis=1)
    layer_num = 2
    all_gates_tensor_sum = tf.zeros([tf.shape(all_gates_tensor)[0],tf.shape(all_gates_tensor)[1]],tf.float32) 
    for i in range(layer_num):
        #[task_num,B*expert_num]
        all_gates_tensor = tf.matmul(Adj,all_gates_tensor)
        all_gates_tensor = tf.nn.leaky_relu(all_gates_tensor)
        all_gates_tensor_sum += all_gates_tensor
    #[task_num,B,expert_num]
    all_gates_output = tf.reshape(all_gates_tensor,[len(task_list),-1,gates_output["task0"].shape[1]])
    #task_num*[B,expert_num]
    split_all_gates = tf.split(all_gates_output,len(task_list),axis=0)
    for task_idx in range(len(split_all_gates)):
        gates_output["task" + str(task_idx)] = tf.squeeze(split_all_gates[task_idx], axis=0)
        gates_output["task" + str(task_idx)] = tf.nn.softmax(gates_output["task" + str(task_idx)],axis=1)
        gates_output["task_new" + str(task_idx)] = tf.identity(gates_output["task" + str(task_idx)], name="task_out"+str(task_idx))
    #每个Gate与共享的多个Expert相乘，Gate是一个概率分布，控制每个Expert对task的贡献程度，比如taskA的gate为(0.1,0.2,0.7)，则代表Expert0、Expert1、Expert2对taskA的贡献程度分别为0.1、0.2和0.7
    final_outputs = {}
    for task_idx in range(len(task_list)):
        weight = tf.expand_dims(gates_output["task_new" + str(task_idx)], axis=1)  # b * 1 * 8
        experts = tf.concat([expert_outputs["expert" + str(expert_idx)] for expert_idx in range(expert_num)], axis=1)# B * 8 * w
        #通过对比学习得到不同专家的向量表示
        #bsz, seqlen, emd_dim = experts.shape
        # compute cl loss
        #归一化处理再相乘的意义
        norm_rep = experts / tf.norm(experts,axis = 2,keepdims = True)#last_hidden_states.norm(dim=2, keepdim=True) ord="fro",
        norm_rep_1 = tf.compat.v1.nn.dropout(norm_rep,keep_prob=0.9)
        norm_rep_2 = tf.compat.v1.nn.dropout(norm_rep,keep_prob=0.9)
        cosine_scores = norm_rep_1 @ tf.compat.v1.matrix_transpose(norm_rep_2) #perm=[0, 2, 1]
        #assert cosine_scores.shape == [bsz, seqlen, seqlen]
        #tmp_matrix = tf.squeeze(experts[:,:,0:1], axis=2)
        #input_ids = tf.ones_like(tmp_matrix, dtype=tf.float32)
        #input_ids = tf.ones([bsz, seqlen],dtype = tf.float32)
        cl_loss = contrastive_loss(cosine_scores)
        cl_loss_sum += cl_loss

        output = tf.matmul(weight, experts)  # b * 1 * w
        #[B,W]
        #output = tf.squeeze(output, axis=1)
        #[0.1 0.2 0.5 0.4 0.2
        # 0.2 0.5 0.3 0.2 0.5
        # 0.5 0.7 0.1 0.4 0.8
        # 0.5.0.7.0.2.0.3 0.6]
        final_outputs["task" + str(task_idx)] = output


    return final_outputs,loss_gates,cl_loss_sum


def combine_feature(features, mode, params):
    feature_indices, feature_values = features['feature_indices'], features['feature_values']

    itemid_indices = tf.gather(params=feature_indices, indices=params['index.albumid'], axis=1)
    itemid_values = tf.gather(params=feature_values, indices=params['index.albumid'], axis=1)

    channelid_indices = tf.gather(params=feature_indices, indices=params['index.channelid'], axis=1)
    channelid_values = tf.gather(params=feature_values, indices=params['index.channelid'], axis=1)

    deepfm_indices = tf.gather(params=feature_indices, indices=params['index.deepfm'], axis=1)
    deepfm_values = tf.gather(params=feature_values, indices=params['index.deepfm'], axis=1)

    click_behavior_itemid_indices = tf.gather(params=feature_indices, indices=params['index.play_album_7d_seq'], axis=1)
    click_behavior_itemid_values = tf.gather(params=feature_values, indices=params['index.play_album_7d_seq'], axis=1)

    click_behavior_channelid_indices = tf.gather(params=feature_indices, indices=params['index.play_channel_7d_seq'], axis=1)
    click_behavior_channelid_values = tf.gather(params=feature_values, indices=params['index.play_channel_7d_seq'], axis=1)

    click_behavior_clicknum_indices = tf.gather(params=feature_indices, indices=params['index.play_cnt_7d_seq'], axis=1)
    click_behavior_clicknum_values = tf.gather(params=feature_values, indices=params['index.play_cnt_7d_seq'], axis=1)

    #deepfm_feature = deepfm(deepfm_indices, deepfm_values, mode, params)

    embedding_matrix = tf.compat.v1.get_variable('embedding_matrix',
                                                 [params['data.feature_size'], params['network.embedding_size']],
                                                 dtype=tf.float32,
                                                 initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1))

    # [B, 1, D]
    itemid_embed = tf.nn.embedding_lookup(embedding_matrix, itemid_indices)
    # [B, SEQ_LENGTH, D]
    itemid_embed_tile = tf.tile(itemid_embed, [1, tf.shape(click_behavior_itemid_indices)[1], 1])
    # [B, 1, D]
    channelid_embed = tf.nn.embedding_lookup(embedding_matrix, channelid_indices)
    # [B, SEQ_LENGTH, D]
    channelid_embed_tile = tf.tile(channelid_embed, [1, tf.shape(click_behavior_itemid_indices)[1], 1])
    # [B, SEQ_LENGTH, D']
    item_embedding = tf.concat([itemid_embed_tile, channelid_embed_tile], axis=2)
    # [B, SEQ_LENGTH, D]
    behavior_itemid_embed = tf.nn.embedding_lookup(embedding_matrix, click_behavior_itemid_indices)
    # [B, SEQ_LENGTH, D]
    behavior_channelid_embed = tf.nn.embedding_lookup(embedding_matrix, click_behavior_channelid_indices)
    # [B, SEQ_LENGTH, D']
    behavior_embedding = tf.concat([behavior_itemid_embed, behavior_channelid_embed], axis=2)
    # [B, SEQ_LENGTH, D]
    content_embedding = tf.nn.embedding_lookup(embedding_matrix, click_behavior_clicknum_indices)
    # [B, 1]
    sequence_length = tf.gather(click_behavior_itemid_values, [0], axis=1)
    # [B, 1, D]
    user_vector = din_layer("itemid_attention", item_embedding, behavior_embedding, sequence_length, content_embedding)

    itemid_inner, itemid_cosin, itemid_diff, itemid_diff_square = cosine_similarity(user_vector, tf.concat([itemid_embed, channelid_embed], axis=-1))

    #[B,D']
    feature = tf.concat([tf.squeeze(user_vector,  axis=1),
                         tf.squeeze(itemid_embed, axis=1),
                         tf.squeeze(channelid_embed, axis=1),
                         tf.squeeze(itemid_inner, axis=1),
                         tf.squeeze(itemid_cosin, axis=1),
                         tf.squeeze(itemid_diff,  axis=1),
                         tf.squeeze(itemid_diff_square, axis=1)], axis=1)

    deepfm_feature,feature_embed = deepfm(deepfm_indices, deepfm_values, mode, params,feature)
    return deepfm_feature,feature_embed


def deepfm(feature_indices, feature_values, mode, params, feature):
    # ============= input =============
    # (feature_indices = feature_values = [batch_size, field_size], labels = [batch_size, 1])
    # ============= init feature embeddings for FM second order part and Deep part =============
    # (feature_embeddings = [batch_size, field_size, embedding_size])

    with tf.compat.v1.variable_scope('deepfm_variables', reuse=tf.compat.v1.AUTO_REUSE):
        embedding_matrix = tf.compat.v1.get_variable('embedding_matrix',
                                                     [params['data.feature_size'], params['network.embedding_size']],
                                                     dtype=tf.float32,
                                                     initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1))
    #[B,F,D]
    feature_embeddings = tf.nn.embedding_lookup(params=embedding_matrix, ids=feature_indices)
    #[B,F,1]
    feature_values = tf.expand_dims(feature_values, axis=-1)
    #[B,F,D]*[B,F,1] 相当于将embedding向量乘以value值
    feature_embeddings = tf.multiply(feature_embeddings, feature_values, name='feature_embeddings')

    # ============= FM part =============
    # FM first order part (first_order_output = [batch_size, 1])
    first_order_matrix = tf.Variable(tf.random.normal([params['data.feature_size'], 1], 0.0, 1.0),
                                     name='first_order_matrix')
    first_order_output = tf.nn.embedding_lookup(params=first_order_matrix, ids=feature_indices)

    first_order_output = tf.reduce_sum(tf.multiply(first_order_output, feature_values), axis=2,
                                       name='first_order_output')

    # FM second order part (second_order_output = [batch_size, 1])
    square_of_sum = tf.square(tf.reduce_sum(feature_embeddings, axis=1, keepdims=None))
    sum_of_square = tf.reduce_sum(tf.square(feature_embeddings), axis=1, keepdims=None)
    second_order_output = tf.multiply(0.5, tf.subtract(square_of_sum, sum_of_square), name='second_order_output')

    # ============= Deep part =============
    #[B,L*D]
    feature_embed = tf.reshape(feature_embeddings, shape=[-1, len(params['index.deepfm']) * params['network.embedding_size']])
    net = tf.concat([feature_embed,feature],axis=1)

    for idx, units in enumerate(params['network.hidden_units'], start=1):
        net = tf.compat.v1.layers.dense(inputs=net, units=units,
                                        kernel_initializer=tf.compat.v1.keras.initializers.he_normal(),
                                        kernel_regularizer=tf.compat.v1.keras.regularizers.l2(0.01),
                                        name='deepfm_dense_layer_{}'.format(idx))
        if params['network.batch_norm']:
            net = tf.compat.v1.layers.batch_normalization(inputs=net, training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                          name='deepfm_bn_layer_{}'.format(idx))
        net = tf.nn.relu(net, name='deepfm_relu_layer_{}'.format(idx))
        if params['network.dropout'] and params['network.dropout_rate'] > 0:
            net = tf.compat.v1.layers.dropout(inputs=net, rate=params['network.dropout_rate'],
                                              training=(mode == tf.estimator.ModeKeys.TRAIN),
                                              name='deepfm_dropout_layer_{}'.format(idx))

    # ============= DeepFM network output =============
    output = tf.concat([first_order_output, second_order_output, net], axis=1)

    return output,feature_embed


def din_layer(feature_tag, item_embed, seq_embed, seq_length, content_embed):

    # [B,SEQ_LENGTH,D']
    attention_layer = tf.concat(
            [item_embed, seq_embed, seq_embed - item_embed,
             seq_embed * item_embed, content_embed], axis=-1)

    attention_layer = tf.compat.v1.layers.dense(attention_layer, 1, None,
                                                kernel_initializer=tf.compat.v1.keras.initializers.glorot_normal(),
                                                name=feature_tag + "attention")  # B * SEQ_LENGTH * 1

    attention_layer = tf.transpose(attention_layer, [0, 2, 1])  # B * 1 * SEQ_LENGTH

    attention_layer = attention_layer / 0.3

    facts_length = tf.reshape(seq_length, [tf.shape(seq_length)[0]])  # [B]

    attention_mask = tf.sequence_mask(facts_length, tf.shape(seq_embed)[1])  # [B, SEQ_LENGTH]

    attention_mask = tf.expand_dims(attention_mask, axis=1)  # [B, 1, SEQ_LENGTH]

    paddings = tf.ones_like(attention_layer) * (-2 ** 32 + 1)

    attention_layer = tf.where(attention_mask, attention_layer, paddings)  # [B, 1, SEQ_LENGTH]

    attention_layer = tf.nn.softmax(attention_layer)

    # [B, 1, D]
    user_vector = tf.matmul(attention_layer, seq_embed)

    return user_vector


def cosine_similarity(user_vector, poi_embedding):
    inner = tf.reduce_sum(user_vector * poi_embedding, axis=2, keepdims=True)  # [B,1,E]
    cosin = inner / tf.expand_dims(
        ((tf.norm(user_vector, ord=2, axis=2) + 0.0000001) * (tf.norm(poi_embedding, ord=2, axis=2) + 0.00000001)),
        -1)
    diff = tf.abs(user_vector - poi_embedding)

    return inner, cosin, diff, diff * diff
