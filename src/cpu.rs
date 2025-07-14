use strum::IntoEnumIterator;
use strum_macros::EnumIter;

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum Instruction {
    // Reg to Reg 64 bit instructions
    Add(Reg, Reg),
    Sub(Reg, Reg),
    Mul(Reg, Reg),
    Div(Reg, Reg),
    Mod(Reg, Reg),
    Pow(Reg, Reg),
    Mov(Reg, Reg),
    StoreReg(Reg, Reg, u64),
    LoadReg(Reg, Reg, u64),
    Cmp(Reg, Reg),
    StoreMem(Reg, u64),
    LoadMem(Reg, u64),
    MovImm(Reg, u64),
    Branch(u64),
    BranchEq(u64),
    BranchNe(u64),
    BranchGt(u64),
    BranchLt(u64),
    PushVal(u64),
    PushReg(Reg),
    Pop,
    PopReg(Reg),
    Call(u64),
    Return,
    Halt,
}

const CATEGORY_MAX: u8 = 0xF0;
const SIZE_MAX: u8 = 0x0F;

// Instruction size
const SIZE_64: u8 = 0x0;
//const SIZE_32: u8 = 0x1;
//const SIZE_16: u8 = 0x2;
//const SIZE_8: u8 = 0x3;

// Instruction category
const REG_REG_INS: u8 = 0x10;
const REG_MEM_INS: u8 = 0x20;
const CTL_FLW_INS: u8 = 0x30;
const STK_INS: u8 = 0x40;
const IMM_INS: u8 = 0x50;
const REG_LAD_STR_INS: u8 = 0x60;

// Instructions
const INS_ADD: u8 = 0x1;
const INS_SUB: u8 = 0x2;
const INS_MUL: u8 = 0x3;
const INS_DIV: u8 = 0x4;
const INS_MOD: u8 = 0x5;
const INS_POW: u8 = 0x6;
const INS_MOV: u8 = 0x7;
const INS_STR: u8 = 0x8;
const INS_LAD: u8 = 0x9;
const INS_CMP: u8 = 0xA;
const INS_BR: u8 = 0xB;
const INS_BREQ: u8 = 0xC;
const INS_BRNE: u8 = 0xD;
const INS_BRLT: u8 = 0xE;
const INS_BRGT: u8 = 0xF;
const INS_PUSH_REG: u8 = 0x10;
const INS_PUSH_VAL: u8 = 0x11;
const INS_POP: u8 = 0x12;
const INS_POP_REG: u8 = 0x13;
const INS_CALL: u8 = 0x14;
const INS_RET: u8 = 0x15;
const INS_HLT: u8 = 0x16;

impl Instruction {
    pub fn deserialize(data: &[u8]) -> Result<(Instruction, usize), String> {
        if data.len() < 2 {
            return Err("Not enough data to decode".into());
        }

        let cat_size = data[0];
        let category = cat_size & CATEGORY_MAX;
        let _size = cat_size & SIZE_MAX;
        let opcode = data[1];

        let mut offset = 2;

        let take_u64 = |data: &[u8], offset: &mut usize| -> Result<u64, String> {
            if data.len() < *offset + 8 {
                return Err("Not enough data for u64".into());
            }
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&data[*offset..*offset + 8]);
            *offset += 8;
            Ok(u64::from_le_bytes(bytes))
        };

        let take_reg = |data: &[u8], offset: &mut usize| -> Result<Reg, String> {
            if data.len() <= *offset {
                return Err("Not enough data for register".into());
            }
            let r = data[*offset];
            *offset += 1;
            Reg::try_from(r).map_err(|_| format!("Invalid register: {}", r))
        };

        let instr = match category {
            REG_REG_INS => {
                let reg1 = take_reg(data, &mut offset)?;
                let reg2 = take_reg(data, &mut offset)?;
                match opcode {
                    INS_ADD => Instruction::Add(reg1, reg2),
                    INS_SUB => Instruction::Sub(reg1, reg2),
                    INS_MUL => Instruction::Mul(reg1, reg2),
                    INS_DIV => Instruction::Div(reg1, reg2),
                    INS_MOD => Instruction::Mod(reg1, reg2),
                    INS_POW => Instruction::Pow(reg1, reg2),
                    INS_MOV => Instruction::Mov(reg1, reg2),
                    INS_CMP => Instruction::Cmp(reg1, reg2),
                    _ => return Err(format!("Invalid REG_REG opcode: {}", opcode)),
                }
            }
            REG_MEM_INS => {
                let reg = take_reg(data, &mut offset)?;
                let addr = take_u64(data, &mut offset)?;
                match opcode {
                    INS_STR => Instruction::StoreMem(reg, addr),
                    INS_LAD => Instruction::LoadMem(reg, addr),
                    _ => return Err(format!("Invalid REG_MEM opcode: {}", opcode)),
                }
            }
            IMM_INS => {
                let reg = take_reg(data, &mut offset)?;
                let val = take_u64(data, &mut offset)?;
                match opcode {
                    INS_MOV => Instruction::MovImm(reg, val),
                    _ => return Err(format!("Invalid IMM opcode: {}", opcode)),
                }
            }
            CTL_FLW_INS => match opcode {
                INS_BR | INS_BREQ | INS_BRNE | INS_BRGT | INS_BRLT | INS_CALL => {
                    let addr = take_u64(data, &mut offset)?;
                    match opcode {
                        INS_BR => Instruction::Branch(addr),
                        INS_BREQ => Instruction::BranchEq(addr),
                        INS_BRNE => Instruction::BranchNe(addr),
                        INS_BRGT => Instruction::BranchGt(addr),
                        INS_BRLT => Instruction::BranchLt(addr),
                        INS_CALL => Instruction::Call(addr),
                        _ => unreachable!(),
                    }
                }
                INS_RET => Instruction::Return,
                INS_HLT => Instruction::Halt,
                _ => return Err(format!("Invalid CTL_FLW opcode: {}", opcode)),
            },
            STK_INS => match opcode {
                INS_PUSH_VAL => {
                    let val = take_u64(data, &mut offset)?;
                    Instruction::PushVal(val)
                }
                INS_PUSH_REG => {
                    let reg = take_reg(data, &mut offset)?;
                    Instruction::PushReg(reg)
                }
                INS_POP => Instruction::Pop,
                INS_POP_REG => {
                    let reg = take_reg(data, &mut offset)?;
                    Instruction::PopReg(reg)
                }
                _ => return Err(format!("Invalid STK_INS opcode: {}", opcode)),
            },
            REG_LAD_STR_INS => match opcode {
                INS_LAD => {
                    let reg1 = take_reg(data, &mut offset)?;
                    let reg2 = take_reg(data, &mut offset)?;
                    let val = take_u64(data, &mut offset)?;
                    Instruction::LoadReg(reg1, reg2, val)
                }
                INS_STR => {
                    let reg1 = take_reg(data, &mut offset)?;
                    let reg2 = take_reg(data, &mut offset)?;
                    let val = take_u64(data, &mut offset)?;
                    Instruction::StoreReg(reg1, reg2, val)
                }
                _ => return Err(format!("Invalid REG_LAD_STR_INS opcode : {}", opcode)),
            },
            _ => return Err(format!("Unknown instruction category: {:#x}", category)),
        };

        Ok((instr, offset))
    }

    pub fn serialize(self: Self) -> Vec<u8> {
        let mut data: Vec<u8> = Vec::new();
        match self {
            Instruction::Add(reg, reg1) => {
                data.extend_from_slice(&[REG_REG_INS | SIZE_64, INS_ADD, reg as u8, reg1 as u8]);
            }
            Instruction::Sub(reg, reg1) => {
                data.extend_from_slice(&[REG_REG_INS | SIZE_64, INS_SUB, reg as u8, reg1 as u8]);
            }
            Instruction::Mul(reg, reg1) => {
                data.extend_from_slice(&[REG_REG_INS | SIZE_64, INS_MUL, reg as u8, reg1 as u8]);
            }
            Instruction::Div(reg, reg1) => {
                data.extend_from_slice(&[REG_REG_INS | SIZE_64, INS_DIV, reg as u8, reg1 as u8]);
            }
            Instruction::Mod(reg, reg1) => {
                data.extend_from_slice(&[REG_REG_INS | SIZE_64, INS_MOD, reg as u8, reg1 as u8]);
            }
            Instruction::Pow(reg, reg1) => {
                data.extend_from_slice(&[REG_REG_INS | SIZE_64, INS_POW, reg as u8, reg1 as u8]);
            }
            Instruction::Mov(reg, reg1) => {
                data.extend_from_slice(&[REG_REG_INS | SIZE_64, INS_MOV, reg as u8, reg1 as u8]);
            }
            Instruction::StoreReg(reg, reg1, offset) => {
                data.extend_from_slice(&[
                    REG_LAD_STR_INS | SIZE_64,
                    INS_STR,
                    reg as u8,
                    reg1 as u8,
                ]);
                data.extend_from_slice(&offset.to_le_bytes());
            }
            Instruction::LoadReg(reg, reg1, offset) => {
                data.extend_from_slice(&[
                    REG_LAD_STR_INS | SIZE_64,
                    INS_LAD,
                    reg as u8,
                    reg1 as u8,
                ]);
                data.extend_from_slice(&offset.to_le_bytes());
            }
            Instruction::Cmp(reg, reg1) => {
                data.extend_from_slice(&[REG_REG_INS | SIZE_64, INS_CMP, reg as u8, reg1 as u8]);
            }
            Instruction::StoreMem(reg, addr) => {
                data.extend_from_slice(&[REG_MEM_INS | SIZE_64, INS_STR, reg as u8]);
                data.extend_from_slice(&addr.to_le_bytes());
            }
            Instruction::LoadMem(reg, addr) => {
                data.extend_from_slice(&[REG_MEM_INS | SIZE_64, INS_LAD, reg as u8]);
                data.extend_from_slice(&addr.to_le_bytes());
            }
            Instruction::MovImm(reg, val) => {
                data.extend_from_slice(&[IMM_INS, INS_MOV, reg as u8]);
                data.extend_from_slice(&val.to_le_bytes());
            }
            Instruction::Branch(addr) => {
                data.extend_from_slice(&[CTL_FLW_INS, INS_BR]);
                data.extend_from_slice(&addr.to_le_bytes());
            }
            Instruction::BranchEq(addr) => {
                data.extend_from_slice(&[CTL_FLW_INS, INS_BREQ]);
                data.extend_from_slice(&addr.to_le_bytes());
            }
            Instruction::BranchNe(addr) => {
                data.extend_from_slice(&[CTL_FLW_INS, INS_BRNE]);
                data.extend_from_slice(&addr.to_le_bytes());
            }
            Instruction::BranchGt(addr) => {
                data.extend_from_slice(&[CTL_FLW_INS, INS_BRGT]);
                data.extend_from_slice(&addr.to_le_bytes());
            }
            Instruction::BranchLt(addr) => {
                data.extend_from_slice(&[CTL_FLW_INS, INS_BRLT]);
                data.extend_from_slice(&addr.to_le_bytes());
            }
            Instruction::PushVal(val) => {
                data.extend_from_slice(&[STK_INS | SIZE_64, INS_PUSH_VAL]);
                data.extend_from_slice(&val.to_le_bytes());
            }
            Instruction::PushReg(reg) => {
                data.extend_from_slice(&[STK_INS | SIZE_64, INS_PUSH_REG, reg as u8]);
            }
            Instruction::Pop => {
                data.extend_from_slice(&[STK_INS | SIZE_64, INS_POP]);
            }
            Instruction::PopReg(reg) => {
                data.extend_from_slice(&[STK_INS | SIZE_64, INS_POP_REG, reg as u8]);
            }
            Instruction::Call(addr) => {
                data.extend_from_slice(&[CTL_FLW_INS, INS_CALL]);
                data.extend_from_slice(&addr.to_le_bytes());
            }
            Instruction::Return => {
                data.extend_from_slice(&[CTL_FLW_INS, INS_RET]);
            }
            Instruction::Halt => {
                data.extend_from_slice(&[CTL_FLW_INS, INS_HLT]);
            }
        }
        data
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
    // Argument 0
    A0 = 5,
    // Argument 1
    A1 = 6,
    // Argument 2
    A2 = 7,
    T0 = 8,
    T1 = 9,
    T2 = 10,
}

impl Reg {
    pub fn argument_registers() -> Vec<Reg> {
        vec![Reg::A0, Reg::A1, Reg::A2]
    }

    pub fn general_purpose_registers() -> Vec<Reg> {
        Self::iter()
            .filter_map(|reg| {
                if (reg != Reg::FL)
                    && (reg != Reg::FP)
                    && (reg != Reg::PC)
                    && (reg != Reg::RA)
                    && (reg != Reg::SP)
                {
                    return Some(reg);
                }
                None
            })
            .collect()
    }
}

impl std::convert::TryFrom<u8> for Reg {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Reg::PC),
            1 => Ok(Reg::FL),
            2 => Ok(Reg::FP),
            3 => Ok(Reg::SP),
            4 => Ok(Reg::RA),
            5 => Ok(Reg::A0),
            6 => Ok(Reg::A1),
            7 => Ok(Reg::A2),
            8 => Ok(Reg::T0),
            9 => Ok(Reg::T1),
            10 => Ok(Reg::T2),
            _ => Err(()),
        }
    }
}
