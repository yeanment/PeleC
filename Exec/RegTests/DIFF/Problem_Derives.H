// problem-specific derived variables would be put here

derive_lst.add("cp",IndexType::TheCellType(),1,pc_dercp,the_same_box);
derive_lst.addComponent("cp",desc_lst,State_Type,Density,NUM_STATE);

derive_lst.add("cv",IndexType::TheCellType(),1,pc_dercv,the_same_box);
derive_lst.addComponent("cv",desc_lst,State_Type,Density,NUM_STATE);

derive_lst.add("wbar",IndexType::TheCellType(),1,pc_derwbar,the_same_box);
derive_lst.addComponent("wbar",desc_lst,State_Type,Density,NUM_STATE);

derive_lst.add("mu",IndexType::TheCellType(),1,pc_dermu,the_same_box);
derive_lst.addComponent("mu",desc_lst,State_Type,Density,NUM_STATE);

derive_lst.add("lam",IndexType::TheCellType(),1,pc_derlam,the_same_box);
derive_lst.addComponent("lam",desc_lst,State_Type,Density,NUM_STATE);
