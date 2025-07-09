use strum_macros::EnumIter;

pub enum Instruction {
    Add(Reg, Reg),
    Sub(Reg, Reg),
    Mul(Reg, Reg),
    Div(Reg, Reg),
    Mod(Reg, Reg),
    Pow(Reg, Reg),
    Mov(Reg, Reg),
    StoreMem(Reg, u64),
    StoreReg(Reg, Reg),
    LoadMem(Reg, u64),
    LoadReg(Reg, Reg),
    Branch(u64),
    BranchEq(u64),
    BranchNe(u64),
    BranchGt(u64),
    BranchLt(u64),
    Cmp(Reg, Reg),
    PushReg(Reg),
    PushVal(u64),
    Pop,
    PopReg(Reg),
    Call(u64),
    Return,
}

impl Instruction {
    pub fn deserialize(data: &[u8]) -> Result<Instruction, String> {
        todo!()
    }

    pub fn serialize(self: Self) -> Vec<u8> {
        todo!()
    }
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, EnumIter)]
#[repr(u8)]
pub enum Reg {
    PC = 0,
    FL = 1,
    FP = 2,
    SP = 3,
    RA = 4,
    A0 = 5,
    A1 = 6,
    A2 = 7,
    T0 = 8,
    T1 = 9,
    T2 = 10,
}
