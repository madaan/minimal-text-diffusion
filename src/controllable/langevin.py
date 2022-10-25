"""
Utilizes a trained classifier model to guide the diffusion process.

- Given:
1. input embeddings
2. A classifier model
3. Labels

The classifier model is used to refine the input embeddings such that the logits of the classifier model are maximized for the labels.
"""
import torch


def langevin_binary_classifier(classifier, label_ids, x_t, t, num_langevin_steps: int = 1, step_size: float=1e-2):  # current best.

    x_t_as_params = torch.nn.Parameter(x_t)

    with torch.enable_grad():
        for i in range(num_langevin_steps):
            optimizer = torch.optim.Adagrad([x_t_as_params], lr=step_size)

            optimizer.zero_grad()
            model_out = classifier.label_logp(inputs_with_added_noise=x_t_as_params, 
                                              labels=label_ids,
                                              t=t)
            loss = -model_out.loss  # logp 
            loss.backward()
            # print(f"{i}> grad norm: {x_t_as_params.grad.data.norm(2)} | loss: {loss}")
            
            optimizer.step()
           

            x_t_as_params = torch.nn.Parameter(x_t_as_params.data.detach())

    return x_t_as_params.data.detach()
