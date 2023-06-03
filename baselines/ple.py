import tensorflow as tf


def din_deepfm_ple(features, mode, params):
    feature_indices, feature_values = features['feature_indices'], features['feature_values']
    #[B,1]
    #获取batch_size内省份的编码信息
    feature,feature_embed = combine_feature(features, mode, params)
    domain_indices = tf.gather(params=feature_indices, indices=params['index.province'], axis=1)
    task_output = ple_modules(feature, params)
    #net = task_output
    task_list = params['network.task_list']    #['sn','gd','gx','jx']
    final_task = []
    for task_idx in range(len(task_list)):
        task_output["task" + str(task_idx)] = tf.expand_dims(task_output["task" + str(task_idx)], axis=1)
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


    return output




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
# the part for contrastive loss
def build_mask_matrix(seqlen, valid_len_list):
    '''
    (1) if a sequence of length 4 contains zero padding token (i.e., the valid length is 4),
    then the loss padding matrix looks like
    [0., 1., 1., 1.],
    [1., 0., 1., 1.],
    [1., 1., 0., 1.],
    [1., 1., 1., 0.]
    (2) if a sequence of length 4 contains 1 padding token (i.e., the valid length is 3),
    then the loss padding matrix looks like
    [0., 1., 1., 0.],
    [1., 0., 1., 0.],
    [1., 1., 0., 0.],
    [0., 0., 0., 0.]
    '''
    res_list = []

    base_mask = tf.ones((seqlen,seqlen),dtype = tf.float32) - tf.eye(seqlen)
    #base_mask = tf.Variable(base_mask,dtype = tf.float32)
    #print(base_mask)
    bsz = len(valid_len_list)
    for i in range(bsz):
        one_base_mask = tf.Variable(base_mask)#[4,4]
        #print(one_base_mask)
        one_valid_len = valid_len_list[i]

        for j in range(one_valid_len,seqlen):
            for k in range(seqlen):
                #print(j)
                one_base_mask[k,j].assign(0.) # = 0.
                one_base_mask[j,k].assign(0.) # = 0.
        #one_base_mask[:,one_valid_len:].assign(0.)# = 0.
        #one_base_mask[one_valid_len:, :].assign(0.)# = 0.
        #print(one_base_mask)
        res_list.append(one_base_mask)

    res_mask = tf.stack(res_list,0)
    #print(res_mask)
    #assert res_mask.shape == tf.shape([bsz, seqlen, seqlen])
    return tf.convert_to_tensor(res_mask)

def contrastive_loss(score_matrix, input_ids):
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
    pad_token_id = 0.0
    ### input mask
    input_mask = tf.ones(input_ids.shape,dtype = tf.float32) #torch.ones_like(input_ids).type(torch.FloatTensor)
    #if loss_matrix.is_cuda:
    #    input_mask = input_mask.cuda(loss_matrix.get_device())
    input_mask = tf.where(input_ids!=pad_token_id,input_mask,0.0)#input_mask.masked_fill(input_ids.eq(self.pad_token_id), 0.0)
    #print(input_mask)
    #print(input_mask.shape)
    #if loss_matrix.is_cuda:
    #    input_mask = input_mask.cuda(loss_matrix.get_device())

    #valid_len_list = torch.sum(input_mask, dim = -1).tolist()
    valid_len_list = tf.reduce_sum(input_mask,axis = -1)
    valid_len_list = list(valid_len_list.numpy())

    loss_mask = build_mask_matrix(seqlen, [int(item) for item in valid_len_list])

    #if score_matrix.is_cuda:
    #    loss_mask = loss_mask.cuda(score_matrix.get_device())
    #将对角线内容置为0
    masked_loss_matrix = loss_matrix * loss_mask

    loss_matrix = tf.reduce_sum(masked_loss_matrix, axis = -1)
    #assert loss_matrix.shape == input_ids.shape
    loss_matrix = loss_matrix * input_mask
    cl_loss = tf.reduce_sum(loss_matrix) / tf.reduce_sum(loss_mask)
    return cl_loss


def ple_modules(features,params):
    #features [B,D']
    task_list = params['network.task_list']    #['sn','gd','gx','jx']
    task_expert_param = params['network.task_expert_param']  # [2,2,2]
    shared_expert_num = params['network.shared_expert_num']  # 2
    expert_hidden_units = params['network.expert_hidden_units']  # [256, 32]
    extraction_layer_num = params['network.extraction_layer_num'] # 2

    expert_inputs = {}

    task_num = len(task_expert_param)
    for idx in range(task_num):
        expert_inputs["task" + str(idx)] = features
    expert_inputs["share"] = features

    for layer_idx in range(extraction_layer_num):
        with tf.compat.v1.variable_scope("extraction_layer_%d" % layer_idx, reuse=tf.compat.v1.AUTO_REUSE):
            expert_inputs = extraction_network(expert_inputs, task_expert_param, shared_expert_num,
                                                   expert_hidden_units, False)

    with tf.compat.v1.variable_scope("extraction_layer_final", reuse=tf.compat.v1.AUTO_REUSE):
        task_output = extraction_network(expert_inputs, task_expert_param, shared_expert_num,
                                             expert_hidden_units, True)

    return task_output

def extraction_network(expert_inputs, task_expert_param, shared_expert_num, expert_hidden_units,
                       is_final_layer):
    """
    expert_inputs: 输入
    task_expert_param: [3,3], 每个task对应的expert数量，从左到右建立
    shared_expert_num: 3,  share对应的expert数量
    extraction_layer: 层数
    ple_expert_units: [256, 128] expert 隐藏层单元
    """
    task_num = len(task_expert_param)

    expert_outputs = {}
    for idx in range(task_num):
        expert_outputs["task" + str(idx)] = []
    expert_outputs["share"] = []

    with tf.compat.v1.variable_scope("experts", reuse=tf.compat.v1.AUTO_REUSE):
        for task_idx, num_experts in enumerate(task_expert_param):
            for expert_idx in range(num_experts):
                y = expert_inputs["task" + str(task_idx)]
                for idx, layer_size in enumerate(expert_hidden_units):
                    scope_name_tmp = "task_%d_expert_%d_layer_%d" % (task_idx, expert_idx, idx)
                    y = tf.compat.v1.layers.dense(inputs=y, units=layer_size,
                                              kernel_initializer=tf.compat.v1.keras.initializers.he_normal(),
                                              kernel_regularizer=tf.compat.v1.keras.regularizers.l2(0.01),
                                              name=scope_name_tmp)
                    y = tf.nn.relu(y, name="task_%d_expert_%d_layer_%d_relu" % (task_idx, expert_idx, idx))

                r = tf.expand_dims(y, axis=1)
                expert_outputs["task" + str(task_idx)].append(r)

    with tf.compat.v1.variable_scope("shares", reuse=tf.compat.v1.AUTO_REUSE):
        for expert_idx in range(shared_expert_num):
            y = expert_inputs["share"]
            for idx, layer_size in enumerate(expert_hidden_units):
                scope_name_tmp = "share_expert_%d_layer_%d" % (expert_idx, idx)
                y = tf.compat.v1.layers.dense(inputs=y, units=layer_size,
                                              kernel_initializer=tf.compat.v1.keras.initializers.he_normal(),
                                              kernel_regularizer=tf.compat.v1.keras.regularizers.l2(0.01),
                                              name=scope_name_tmp)
                y = tf.nn.relu(y, name="share_%d_expert_%d_layer_%d_relu" % (task_idx, expert_idx, idx))
            r = tf.expand_dims(y, axis=1)
            expert_outputs["share"].append(r)

    gates_output = {}
    with tf.compat.v1.variable_scope("gates_expert", reuse=tf.compat.v1.AUTO_REUSE):
        for task_idx, num_experts in enumerate(task_expert_param):
            scope_name_tmp = "gates_task_%d" % (task_idx)
            y = expert_inputs["task" + str(task_idx)]
            z = tf.compat.v1.layers.dense(inputs=y, units=(num_experts + shared_expert_num),
                                      kernel_initializer=tf.compat.v1.keras.initializers.he_normal(),
                                      kernel_regularizer=tf.compat.v1.keras.regularizers.l2(0.01),
                                      name=scope_name_tmp)
            z = tf.nn.softmax(z)
            gates_output["task" + str(task_idx)] = z

    if not is_final_layer:
        with tf.compat.v1.variable_scope("gates_share", reuse=tf.compat.v1.AUTO_REUSE):
            total_num_experts = shared_expert_num
            for task_idx, num_experts in enumerate(task_expert_param):
                total_num_experts += num_experts
            scope_name_tmp = "gates_share"
            y = expert_inputs["share"]
            z = tf.compat.v1.layers.dense(inputs=y, units=total_num_experts,
                                      kernel_initializer=tf.compat.v1.keras.initializers.he_normal(),
                                      kernel_regularizer=tf.compat.v1.keras.regularizers.l2(0.01),
                                      name=scope_name_tmp)
            z = tf.nn.softmax(z)
            gates_output["share"] = z

    final_outputs = {}
    for task_idx in range(task_num):
        weight = tf.expand_dims(gates_output["task" + str(task_idx)], axis=1)  # b * 1 * w
        experts = tf.concat(expert_outputs["task" + str(task_idx)] + expert_outputs["share"], axis=1)  # b * w * f
        output = tf.matmul(weight, experts)  # b * 1 * f
        output = tf.squeeze(output, axis=1)
        final_outputs["task" + str(task_idx)] = output

    if not is_final_layer:
        experts = []
        for task_idx in range(task_num):
            experts = experts + expert_outputs["task" + str(task_idx)]
        experts = experts + expert_outputs["share"]
        weight = tf.expand_dims(gates_output["share"], axis=1)  # b * 1 * w
        experts = tf.concat(experts, axis=1)  # b * w * f
        output = tf.matmul(weight, experts)
        output = tf.squeeze(output, axis=1)
        final_outputs["share"] = output

    return final_outputs



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
